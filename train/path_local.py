# -*- coding: utf-8 -*-
"""Path-local classifier built on UNI2-h embeddings with a FiLM head."""

import os, json, warnings, re
from pathlib import Path
from typing import Dict, List, Tuple
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # keep multi-process h5py stable
# Optional: os.environ.setdefault("OMP_NUM_THREADS", "8")

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from Nuclass.utils.torchvision_compat import ensure_torchvision_nms

ensure_torchvision_nms()
from torchvision import transforms
import timm
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from transformers import logging as hf_logging

try:
    from Nuclass.train.config_utils import (
        resolve_data_dirs,
        resolve_results_dir,
        get_checkpoint_dir,
    )
    from Nuclass.train.config_loader import load_experiment_config
    from Nuclass.train.label_mappings import NEW_LABEL_MAPPING, CLASSES_TO_REMOVE
    from Nuclass.train.dataset_defaults import DEFAULT_DATASET_FOLDERS
except ModuleNotFoundError:
    from config_utils import (
        resolve_data_dirs,
        resolve_results_dir,
        get_checkpoint_dir,
    )
    from config_loader import load_experiment_config
    from label_mappings import NEW_LABEL_MAPPING, CLASSES_TO_REMOVE
    from dataset_defaults import DEFAULT_DATASET_FOLDERS
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*No device id is provided.*", category=UserWarning)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "path_local.yaml"


def prepare_config():
    config, _ = load_experiment_config(
        default_config_path=DEFAULT_CONFIG_PATH,
        description="Path-local training (UNI + FiLM head)",
    )
    config["data_dirs"] = resolve_data_dirs(DEFAULT_DATASET_FOLDERS)
    config["inference_out_dir"] = resolve_results_dir(config["experiment_name"])
    config["checkpoint_dir"] = get_checkpoint_dir(config["experiment_name"])
    return config

# ===============================
# Utils
# ===============================
def assert_cuda_arch_compat():
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 9:
        m = re.match(r"^(\d+)\.(\d+)", torch.__version__)
        ver = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        if ver < (2, 1):
            raise SystemExit(
                f"\n[ERROR] Detected sm_{major}{minor} GPU but PyTorch {torch.__version__} lacks sm_90. "
                f"Install PyTorch >= 2.1 CUDA12 build."
            )

def build_histopath_transforms():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_unnorm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf, val_unnorm

def effective_number_weights(labels: List[int], num_classes: int, beta: float) -> torch.Tensor:
    counts = pd.Series(labels).value_counts().reindex(range(num_classes), fill_value=0).values.astype(np.int64)
    mask = counts > 0
    w = np.zeros(num_classes, dtype=np.float32)
    if mask.any():
        eff_num = 1.0 - np.power(beta, counts[mask])
        w_pos = (1.0 - beta) / np.maximum(eff_num, 1e-8)
        w_pos = w_pos / w_pos.sum() * mask.sum()  # normalize across observed classes only
        w[mask] = w_pos
    return torch.tensor(w, dtype=torch.float)

# ===============================
# Dataset
# ===============================
class AOnlyDataset(Dataset):
    def __init__(self, records: List[Dict], h5_path_224: str,
                 transform_224, val_transform_224_unnorm):
        self.records = records
        self.h5_path = h5_path_224
        self.transform_224 = transform_224
        self.val_transform_unnorm = val_transform_224_unnorm
        self.h5_file = None
        self.is_train = os.path.basename(os.path.dirname(self.h5_path)) == "train"
        self._merged = False
        self._key_to_idx = None

    def __len__(self): return len(self.records)

    def _open(self):
        if self.h5_file is None:
            f = h5py.File(self.h5_path, "r")
            self.h5_file = f
            if "images" in f and "keys" in f:
                self._merged = True
                keys_raw = f["keys"][()]
                keys = [k.decode() if isinstance(k, (bytes, np.bytes_)) else str(k) for k in keys_raw]
                self._key_to_idx = {k: i for i, k in enumerate(keys)}
            else:
                self._merged = False
                self._key_to_idx = None

    def __getitem__(self, idx):
        self._open()
        r = self.records[idx]
        cid = str(r["cell_id"])
        if self._merged:
            i = self._key_to_idx.get(cid)
            if i is None:
                raise KeyError(f"[{self.h5_path}] key not found in merged H5: {cid}")
            img_np = self.h5_file["images"][i][()]
        else:
            if cid not in self.h5_file:
                raise KeyError(f"[{self.h5_path}] key not found: {cid}")
            img_np = self.h5_file[cid][()]

        # standardize to uint8 HxWx3
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[-1] == 4:
            img_np = img_np[..., :3]
        elif img_np.ndim == 3 and img_np.shape[-1] == 3:
            pass
        else:
            raise ValueError(f"Unexpected image shape for {cid}: {img_np.shape}")
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        if img_np.shape != (224, 224, 3):
            raise ValueError(f"[{self.h5_path}] unexpected shape for {cid}: {img_np.shape}")

        x224 = self.transform_224(img_np)
        y = torch.tensor(r["label"], dtype=torch.long)
        tissue = torch.tensor(r["tissue_id"], dtype=torch.long)

        if self.is_train:
            return x224, y, tissue
        x_un = self.val_transform_unnorm(img_np)
        return x224, y, tissue, x_un, cid

# ===============================
# DataModule
# ===============================
class AOnlyDataModule(LightningDataModule):
    def __init__(self, config, label_mapping, classes_to_remove):
        super().__init__()
        self.config = config
        self.label_mapping = label_mapping
        self.classes_to_remove = classes_to_remove
        self.transform_224_train, self.transform_224_val, self.val_transform_unnorm = build_histopath_transforms()

    @staticmethod
    def _norm_tissue(x): return str(x).strip().lower()

    def _preflight_check(self, all_records):
        for root in self.config['data_dirs']:
            if not os.path.isdir(root):
                raise FileNotFoundError(f"[DATA ROOT MISSING] {root}")
            for split in ['train', 'test']:   # your 'test' is val
                meta = os.path.join(root, split, 'annotations.json')
                if not os.path.exists(meta):
                    warnings.warn(f"[WARN] missing {meta}")
                    continue
                h5p = os.path.join(root, split, 'patches_224x224.h5')
                if not os.path.exists(h5p):
                    raise FileNotFoundError(f"[H5 MISSING] {h5p}")
                recs = [r for r in all_records if r['data_dir'] == root and r['split'] == split]
                if not recs:
                    continue
                with h5py.File(h5p, 'r') as f:
                    if "images" in f and "keys" in f:
                        keys_raw = f["keys"][()]
                        keys = {k.decode() if isinstance(k, (bytes, np.bytes_)) else str(k) for k in keys_raw}
                        miss = next((str(r['cell_id']) for r in recs if str(r['cell_id']) not in keys), None)
                        if miss is not None:
                            raise KeyError(f"[{h5p}] missing key in merged H5: {miss}")
                    else:
                        miss = next((str(r['cell_id']) for r in recs if str(r['cell_id']) not in f), None)
                        if miss is not None:
                            raise KeyError(f"[{h5p}] missing key: {miss}")

    def setup(self, stage=None):
        records = []
        for root in self.config['data_dirs']:
            for split in ['train', 'test']:
                meta = os.path.join(root, split, 'annotations.json')
                if not os.path.exists(meta):
                    print(f"[WARN] {meta} not found")
                    continue
                df = pd.read_json(meta, orient='index').reset_index().rename(columns={'index': 'cell_id'})
                df['cell_id'] = df['cell_id'].astype(str)
                if 'tissue_type' not in df.columns:
                    raise KeyError(f"[ERROR] {meta} missing 'tissue_type' field")
                df['tissue_type'] = df['tissue_type'].apply(self._norm_tissue)
                df['split'] = split
                df['data_dir'] = root
                records.extend(df.to_dict('records'))

        self.tissue_names = sorted({r['tissue_type'] for r in records})
        self.tissue_to_idx = {name: i for i, name in enumerate(self.tissue_names)}
        self.num_tissues = len(self.tissue_names)
        print(f"Tissues: {self.tissue_to_idx}")

        filtered = [r for r in records if r.get('class_id') not in self.classes_to_remove]
        mapped = []
        for r in filtered:
            final = self.label_mapping.get(r.get('class_id'))
            if final:
                r['final_class'] = final
                mapped.append(r)

        self.class_names = sorted({r['final_class'] for r in mapped})
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for r in mapped:
            r['label'] = self.class_to_idx[r['final_class']]
            r['tissue_id'] = self.tissue_to_idx[r['tissue_type']]
            r['cell_id'] = str(r['cell_id'])

        self.train_records = [r for r in mapped if r['split'] == 'train']
        self.val_records   = [r for r in mapped if r['split'] == 'test']
        print(f"Classes: {self.num_classes}, Train: {len(self.train_records)}, Val: {len(self.val_records)}")

        labels = [r['label'] for r in self.train_records]
        self.class_weights = effective_number_weights(labels, self.num_classes, self.config["cb_beta"]) if labels else torch.ones(self.num_classes)

        self._preflight_check(self.train_records + self.val_records)

        self.train_sets, self.val_sets = [], []
        for root in self.config['data_dirs']:
            tr = [r for r in self.train_records if r['data_dir'] == root]
            va = [r for r in self.val_records if r['data_dir'] == root]
            if tr:
                self.train_sets.append(AOnlyDataset(
                    tr, os.path.join(root, 'train', 'patches_224x224.h5'),
                    transform_224=self.transform_224_train,
                    val_transform_224_unnorm=None
                ))
            if va:
                self.val_sets.append(AOnlyDataset(
                    va, os.path.join(root, 'test', 'patches_224x224.h5'),
                    transform_224=self.transform_224_val,
                    val_transform_224_unnorm=self.val_transform_unnorm
                ))

        if not self.train_sets:
            raise RuntimeError("Empty train_sets")
        if not self.val_sets:
            warnings.warn("Empty val_sets")

    def _make_loader(self, ds, shuffle):
        # per-device batch; effective batch = per_device_batch * accumulate
        per_device_batch = max(1, self.config['batch_size'] // max(1, self.config.get('accumulate_grad_batches', 1)))
        nw = int(self.config['num_workers'])
        ctx = mp.get_context("spawn")  # avoid fork+h5py issues
        return DataLoader(
            ConcatDataset(ds),
            batch_size=per_device_batch,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=False,        # reset workers to avoid h5 handle leaks
            prefetch_factor=2,               # keep memory usage bounded
            multiprocessing_context=ctx
        )

    def train_dataloader(self): return self._make_loader(self.train_sets, shuffle=True)
    def val_dataloader(self):   return self._make_loader(self.val_sets, shuffle=False)
    def test_dataloader(self):  return self.val_dataloader()

# ===============================
# Model
# ===============================
class FiLM(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int):
        super().__init__()
        hidden = max(64, in_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * feat_dim)
        )

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.net(t)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta

class AOnlyModule(LightningModule):
    def __init__(self, config, num_classes, class_names, num_tissues, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.class_names = class_names
        self.num_classes = num_classes

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # UNI2-h backbone (forward() -> [B,1536])
        uni_cfg = {
            'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
            'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667 * 2,
            'num_classes': 0, 'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
            'reg_tokens': 8, 'dynamic_img_size': True
        }
        self.uni_encoder = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)
        if hasattr(self.uni_encoder, "set_grad_checkpointing"):
            self.uni_encoder.set_grad_checkpointing(True)  # reduce activation memory
        self.uni_feature_dim = self.uni_encoder.embed_dim

        # tissue FiLM adaptor
        tdim = self.hparams.config['tissue_embedding_dim']
        self.tissue_embedder = nn.Embedding(num_tissues, tdim)
        self.tissue_film = FiLM(tdim, self.uni_feature_dim)

        # head
        self.feat_norm = nn.LayerNorm(self.uni_feature_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.uni_feature_dim, 1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

        # loss
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.ones(num_classes))
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=self.hparams.config["label_smoothing"]
        )

        self.val_probs, self.val_labels = [], []
        self.backbone_frozen = False

    # features for probe export
    def extract_features(self, img224, tissues):
        feat = self.uni_encoder(img224)                   # [B, 1536]
        feat = self.feat_norm(feat)
        t = self.tissue_embedder(tissues)                 # [B, tdim]
        gamma, beta = self.tissue_film(t)                 # [B, C], [B, C]
        feat = feat * (1 + gamma) + beta                  # FiLM
        return feat

    def forward(self, img224, tissues):
        feat = self.extract_features(img224, tissues)
        logits = self.head(feat)
        return logits

    def _maybe_freeze_backbone(self, freeze: bool):
        if freeze and not self.backbone_frozen:
            for p in self.uni_encoder.parameters(): p.requires_grad = False
            self.backbone_frozen = True
        if not freeze and self.backbone_frozen:
            for p in self.uni_encoder.parameters(): p.requires_grad = True
            self.backbone_frozen = False

    def on_train_epoch_start(self):
        self._maybe_freeze_backbone(self.current_epoch < self.hparams.config["freeze_backbone_epochs"])

    def training_step(self, batch, batch_idx):
        x224, y, tissue = batch
        logits = self(x224, tissue)
        loss = self.criterion(logits, y)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        return loss

    @staticmethod
    def _conf_acc_count_figure(probs_np, labels_np, title):
        conf = probs_np.max(axis=1)
        preds = probs_np.argmax(axis=1)
        correct = (preds == labels_np).astype(np.float32)
        edges = np.linspace(0.1, 1.0, 10)
        centers = (edges[:-1] + edges[1:]) / 2.0
        counts, accs = [], []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            m = (conf >= lo) & ((conf <= hi) if i == len(edges) - 2 else (conf < hi))
            c = int(m.sum()); counts.append(c)
            accs.append(float(correct[m].mean()) if c > 0 else 0.0)
        fig, ax1 = plt.subplots(figsize=(8, 4.8)); ax2 = ax1.twinx()
        ax1.bar(centers, counts, width=0.08, alpha=0.7, edgecolor='black', label='count')
        ax2.plot(centers, accs, marker='o', label='accuracy')
        ax1.set_xlabel("Confidence"); ax1.set_ylabel("Sample Count"); ax2.set_ylabel("Accuracy")
        ax1.set_xticks(np.round(centers, 1)); ax1.set_title(title)
        ax1.legend(loc='upper left'); ax2.legend(loc='lower right')
        fig.tight_layout()
        return fig

    def _safe_log_image(self, name: str, fig):
        # compatible with WandB / other loggers; swallow logging errors
        try:
            if hasattr(self.logger, "log_image"):
                self.logger.log_image(key=name, images=[fig])
            elif hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
                # WandB style
                self.logger.experiment.log({name: [fig]})
        except Exception:
            pass

    def validation_step(self, batch, batch_idx):
        x224, y, tissue, *_ = batch
        logits = self(x224, tissue)
        probs = torch.softmax(logits, dim=1)
        self.val_probs.append(probs.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        loss = self.criterion(logits, y)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_probs: return
        probs = torch.cat(self.val_probs).numpy()
        labels = torch.cat(self.val_labels).numpy()
        preds = probs.argmax(axis=1)

        acc = accuracy_score(labels, preds)
        f1m = f1_score(labels, preds, average='macro', zero_division=0)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1_macro", f1m, prog_bar=True)

        # visualization hooks
        per_f1 = f1_score(labels, preds, average=None, labels=np.arange(self.num_classes), zero_division=0)
        cm = confusion_matrix(labels, preds, normalize='true')

        if getattr(self.trainer, "is_global_zero", True):
            try:
                fig1, ax1 = plt.subplots(figsize=(max(10, self.num_classes * 0.6), 4))
                sns.barplot(x=self.class_names, y=per_f1, ax=ax1)
                ax1.set_ylim(0, 1); ax1.set_ylabel("F1"); ax1.set_title(f"Per-class F1 (Epoch {self.current_epoch})")
                ax1.tick_params(axis='x', labelrotation=45); plt.setp(ax1.get_xticklabels(), ha='right')
                plt.tight_layout(); self._safe_log_image("val/f1_per_class_bar", fig1); plt.close(fig1)

                fig3, ax3 = plt.subplots(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
                            xticklabels=self.class_names, yticklabels=self.class_names, ax=ax3)
                ax3.set_title(f"Validation Confusion Matrix (Epoch {self.current_epoch})")
                ax3.tick_params(axis='x', labelrotation=45); plt.setp(ax3.get_xticklabels(), ha='right')
                plt.tight_layout(); self._safe_log_image("val/confusion_matrix", fig3); plt.close(fig3)

                fig4 = self._conf_acc_count_figure(probs, labels,
                        title=f"Confidence vs Accuracy/Count (Epoch {self.current_epoch})")
                self._safe_log_image("val/confidence_accuracy_count", fig4); plt.close(fig4)
            except Exception:
                pass

        self.val_probs.clear(); self.val_labels.clear()

    # ---- optimizer with LLRD groups ----
    def configure_optimizers(self):
        wd = self.hparams.config['weight_decay']
        lr_uni = self.hparams.config['lr_uni']
        lr_head = self.hparams.config['lr_head']
        gamma = self.hparams.config['lrd_gamma']

        param_groups = []

        from torch.nn import Parameter
        def add_param_or_module(obj, lr):
            if obj is None:
                return
            if isinstance(obj, Parameter):
                if obj.requires_grad:
                    param_groups.append({"params": [obj], "lr": lr, "weight_decay": wd})
            elif isinstance(obj, (list, tuple)):
                for o in obj:
                    add_param_or_module(o, lr)
            else:
                for p in obj.parameters(recurse=True):
                    if p.requires_grad:
                        param_groups.append({"params": [p], "lr": lr, "weight_decay": wd})

        # collect ViT blocks
        vit_blocks = []
        if hasattr(self.uni_encoder, 'blocks'):
            vit_blocks = list(self.uni_encoder.blocks)
        elif hasattr(self.uni_encoder, 'stages'):
            for s in self.uni_encoder.stages:
                vit_blocks.extend(list(getattr(s, 'blocks', [])))
        n = len(vit_blocks)
        low_lr = lr_uni * (gamma ** n)

        # early components
        add_param_or_module(getattr(self.uni_encoder, 'patch_embed', None), low_lr)
        add_param_or_module(getattr(self.uni_encoder, 'pos_embed', None),  low_lr)
        add_param_or_module(getattr(self.uni_encoder, 'pos_drop',  None),  low_lr)
        add_param_or_module(getattr(self.uni_encoder, 'cls_token',  None),  low_lr)
        add_param_or_module(getattr(self.uni_encoder, 'reg_token',  None),  low_lr)

        # transformer blocks with LLRD
        for i, blk in enumerate(vit_blocks):
            lr = lr_uni * (gamma ** (n - i - 1))
            add_param_or_module(blk, lr)

        # backbone final norm
        add_param_or_module(getattr(self.uni_encoder, 'norm', None), lr_uni)

        # head + tissue adaptor + feature norm
        add_param_or_module(self.head, lr_head)
        add_param_or_module(self.tissue_embedder, lr_head)
        add_param_or_module(self.tissue_film, lr_head)
        add_param_or_module(self.feat_norm, lr_head)

        optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, self.hparams.config["warmup_steps"]))
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# ===============================
# Inference utility
# ===============================
@torch.no_grad()
def run_inference_and_dump(model_ckpt_path, config, dm: AOnlyDataModule,
                           class_names: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AOnlyModule.load_from_checkpoint(
        checkpoint_path=model_ckpt_path,
        config=config,
        num_classes=len(class_names),
        class_names=class_names,
        num_tissues=dm.num_tissues,
        class_weights=dm.class_weights,
        strict=False,
        map_location=device,
    )
    model.to(device).eval()

    # persist mappings
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    with open(os.path.join(out_dir, "class_to_idx.json"), "w") as f:
        json.dump({n: i for i, n in enumerate(class_names)}, f, indent=2)
    with open(os.path.join(out_dir, "tissue_names.json"), "w") as f:
        json.dump(dm.tissue_names, f, indent=2)
    with open(os.path.join(out_dir, "tissue_to_idx.json"), "w") as f:
        json.dump(dm.tissue_to_idx, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    val_loader = dm.val_dataloader()
    all_logits, all_probs, all_labels = [], [], []
    all_embeds, all_cellids = [], []

    for batch in val_loader:
        # x224, y, tissue, x_un, cid
        x224, y, tissue, *_tail = batch
        cid = _tail[-1] if len(_tail) > 0 else None

        x224 = x224.to(device); tissue = tissue.to(device)
        emb = model.extract_features(x224, tissue)
        logits = model.head(emb)
        probs = torch.softmax(logits, dim=1)

        all_embeds.append(emb.cpu())
        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(y)
        if cid is not None:
            if isinstance(cid, torch.Tensor):
                cid = [c for c in cid]
            all_cellids.extend(list(cid))

    embeddings_np = torch.cat(all_embeds).numpy()
    logits_np = torch.cat(all_logits).numpy()
    probs_np  = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    preds_np  = probs_np.argmax(axis=1)

    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings_np)
    np.save(os.path.join(out_dir, "probe.npy"),       embeddings_np)  # alias
    np.save(os.path.join(out_dir, "logits.npy"),      logits_np)
    np.save(os.path.join(out_dir, "probs.npy"),       probs_np)
    np.save(os.path.join(out_dir, "labels.npy"),      labels_np)
    np.save(os.path.join(out_dir, "preds.npy"),       preds_np)
    if len(all_cellids) > 0:
        np.save(os.path.join(out_dir, "cell_ids.npy"), np.array(all_cellids, dtype=object))

    metrics = {"acc": float(accuracy_score(labels_np, preds_np)),
               "f1_macro": float(f1_score(labels_np, preds_np, average="macro", zero_division=0))}
    try:
        metrics["auroc_macro_ovr"] = float(roc_auc_score(labels_np, probs_np, multi_class='ovr', average='macro'))
    except Exception:
        metrics["auroc_macro_ovr"] = None
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[A-only] Inference dumps saved to: {out_dir}")
    print(f"[A-only] Metrics: {metrics}")

# ===============================
# Entry
# ===============================
def main():
    config = prepare_config()

    # enforce DataLoader spawn context for h5py safety
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    seed_everything(config["seed"], workers=True)
    torch.set_float32_matmul_precision('medium')
    assert_cuda_arch_compat()

    # single GPU selection
    if torch.cuda.is_available():
        idx = config["gpus"][0]
        n = torch.cuda.device_count()
        assert 0 <= idx < n, f"GPU index {idx} out of range 0..{n-1}"
        torch.cuda.set_device(idx)
        accelerator, devices, precision = 'gpu', 1, 'bf16-mixed'
    else:
        accelerator, devices, precision = 'cpu', 1, '32'

    # data
    dm = AOnlyDataModule(config, NEW_LABEL_MAPPING, CLASSES_TO_REMOVE)
    dm.setup()

    # model
    model = AOnlyModule(
        config=config,
        num_classes=dm.num_classes,
        class_names=dm.class_names,
        num_tissues=dm.num_tissues,
        class_weights=dm.class_weights
    )

    # logging & callbacks
    wandb_logger = None
    try:
        import pytorch_lightning.loggers as pl_loggers
        wandb_logger = pl_loggers.WandbLogger(
            name=config["experiment_name"],
            project=config["project_name"],
            entity=config["entity"],
            config=config,
            resume='allow'
        )
    except Exception:
        print("[WARN] WandB logger not available")

    ckpt_cb = ModelCheckpoint(
        monitor="val/f1_macro",
        mode="max",
        dirpath=config["checkpoint_dir"],
        filename="best-epoch={epoch}-val_f1={val/f1_macro:.4f}",
    )
    early_cb = EarlyStopping(monitor="val/f1_macro", patience=config["early_stopping_patience"],
                             mode="max", verbose=True)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy='auto',
        precision=precision,
        max_epochs=config["max_epochs"],
        val_check_interval=config["val_check_interval"],
        gradient_clip_val=1.0,
        callbacks=[TQDMProgressBar(refresh_rate=10), early_cb, ckpt_cb],
        logger=wandb_logger,
        log_every_n_steps=10,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=dm)

    best = ckpt_cb.best_model_path
    if best and os.path.exists(best):
        print(f"[A-only] Best checkpoint: {best}")
        run_inference_and_dump(best, config, dm, dm.class_names, config["inference_out_dir"])
    else:
        print("[A-only] No best checkpoint found; skipping inference dump.")

    # cleanup
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    print(f"Done: {config['experiment_name']}")

if __name__ == "__main__":
    main()
