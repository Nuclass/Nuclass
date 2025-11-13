# -*- coding: utf-8 -*-
"""Path-global model (UNI backbone + DINO context) for the released experiments."""

import os, json, re, time, warnings, hashlib, pickle, multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional

# ---- Environment tweaks (before heavy imports) ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "TRUE")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TORCHINDUCTOR_BENCHMARK", "0")  # avoid triton benchmarking stalls

try:
    from Nuclass.train.config_utils import (
        resolve_data_dirs,
        resolve_results_dir,
        get_checkpoint_dir,
        resolve_checkpoint_path,
    )
    from Nuclass.train.config_loader import load_experiment_config
    from Nuclass.train.label_mappings import NEW_LABEL_MAPPING, CLASSES_TO_REMOVE
    from Nuclass.train.dataset_defaults import DEFAULT_DATASET_FOLDERS
except ModuleNotFoundError:
    from config_utils import (
        resolve_data_dirs,
        resolve_results_dir,
        get_checkpoint_dir,
        resolve_checkpoint_path,
    )
    from config_loader import load_experiment_config
    from label_mappings import NEW_LABEL_MAPPING, CLASSES_TO_REMOVE
    from dataset_defaults import DEFAULT_DATASET_FOLDERS

# ===============================
# Config
# ===============================
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "path_global.yaml"
CONFIG = None


def prepare_config():
    global CONFIG
    if CONFIG is not None:
        return CONFIG

    config, _ = load_experiment_config(
        default_config_path=DEFAULT_CONFIG_PATH,
        description="Path-global training (UNI + DINO context)",
    )
    config["data_dirs"] = resolve_data_dirs(DEFAULT_DATASET_FOLDERS)
    config["inference_out_dir"] = resolve_results_dir(config["experiment_name"])
    config["checkpoint_dir"] = get_checkpoint_dir(config["experiment_name"])
    rel = config.pop("teacher_ckpt_rel", None)
    if rel:
        config["teacher_ckpt"] = resolve_checkpoint_path(rel)
    CONFIG = config
    return CONFIG

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from Nuclass.utils.torchvision_compat import ensure_torchvision_nms

ensure_torchvision_nms()
import timm
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

try:
    from Nuclass.train.config_utils import (
        resolve_data_dirs,
        resolve_results_dir,
        get_checkpoint_dir,
        resolve_checkpoint_path,
    )
except ModuleNotFoundError:
    from config_utils import (
        resolve_data_dirs,
        resolve_results_dir,
        get_checkpoint_dir,
        resolve_checkpoint_path,
    )

from transformers import AutoModel, AutoConfig, logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*No device id is provided.*", category=UserWarning)

# ===============================
# Utils
# ===============================
def log_stage(msg: str, t0: float):
    dt = time.perf_counter() - t0
    print(f"[{dt:6.2f}s] {msg}", flush=True)

def assert_cuda_arch_compat():
    if not torch.cuda.is_available(): return
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 9:
        m = re.match(r"^(\d+)\.(\d+)", torch.__version__)
        ver = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        if ver < (2, 1):
            raise SystemExit(f"Detected sm_{major}{minor} but torch {torch.__version__} lacks sm_90. Install >=2.1 CUDA12.")

def effective_number_weights(labels: List[int], num_classes: int, beta: float) -> torch.Tensor:
    counts = pd.Series(labels).value_counts().reindex(range(num_classes), fill_value=0).values.astype(np.int64)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)

# ---- H5 helpers / index cache ----
def _h5_open_read(path: str):
    cfg = prepare_config()
    return h5py.File(
        path, mode="r", libver="latest",
        swmr=bool(cfg.get("h5_open_swmr", True)),
        rdcc_nbytes=int(cfg.get("h5_rdcc_nbytes", 512*1024*1024)),
        rdcc_nslots=int(cfg.get("h5_rdcc_nslots", 1_048_579)),
        rdcc_w0=float(cfg.get("h5_rdcc_w0", 0.75)),
    )

def _file_sig(path: str) -> str:
    st = os.stat(path); return hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()

def _ensure_cache_dir() -> str:
    cfg = prepare_config()
    d = os.path.expanduser(cfg.get("h5_index_cache_dir", "~/.cache/pathb_h5"))
    os.makedirs(d, exist_ok=True); return d

def _load_or_build_k2i(h5_path: Optional[str]) -> Optional[Dict[str, int]]:
    if not h5_path or (not os.path.exists(h5_path)): return None
    cfg = prepare_config()
    cache_on = bool(cfg.get("use_h5_index_cache", True))
    cache_file = None
    if cache_on:
        cache_file = os.path.join(_ensure_cache_dir(), f"{_file_sig(h5_path)}.k2i.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f: return pickle.load(f)
            except Exception: pass
    t0 = time.perf_counter()
    with _h5_open_read(h5_path) as f:
        if "keys" in f:
            try: keys = f["keys"].asstr()[()]
            except Exception:
                raw = f["keys"][()]; keys = [k.decode() if isinstance(k,(bytes,np.bytes_)) else str(k) for k in raw]
            k2i = {str(k): int(i) for i,k in enumerate(keys)}
        else:
            k2i = None
    log_stage(f"build k2i for {os.path.basename(h5_path)}", t0)
    if cache_on and cache_file and k2i is not None:
        try:
            with open(cache_file,"wb") as f: pickle.dump(k2i,f,pickle.HIGHEST_PROTOCOL)
        except Exception: pass
    return k2i

def _maybe_ctx_aligns_with_224(h5_ctx: str, k2i_224: Optional[Dict[str,int]], samples:int=4) -> bool:
    if k2i_224 is None or not os.path.exists(h5_ctx): return False
    try:
        with _h5_open_read(h5_ctx) as f:
            if "keys" not in f: return False
            ds = f["keys"]; N = int(ds.shape[0])
            idxs = [0, N//3, (2*N)//3, N-1] if N>3 else list(range(N))
            hit = 0
            for i in idxs[:max(1, samples)]:
                try: k = ds.asstr()[i]
                except Exception:
                    raw = ds[i]; k = raw.decode() if isinstance(raw,(bytes,np.bytes_)) else str(raw)
                if k2i_224.get(str(k), -1) == i: hit += 1
            return hit >= max(1, len(idxs[:samples]) - 1)
    except Exception:
        return False

# ===============================
# Dataset
# ===============================
class PathBDataset(Dataset):
    def __init__(self, records, h5_224, h5_ctx, ctx_target_size: int,
                 k2i224: Optional[Dict[str,int]], k2iCTX: Optional[Dict[str,int]],
                 ctx_read_mode: str, num_classes: int,
                 resize_on_cpu: bool=False):
        self.records = records
        self.h5_224_path = h5_224
        self.h5_ctx_path = h5_ctx
        self.ctx_target = int(ctx_target_size)
        self.k2i224 = k2i224
        self.k2iCTX = k2iCTX
        self.ctx_read_mode = str(ctx_read_mode).lower()
        self.num_classes = int(num_classes)
        self.resize_on_cpu = bool(resize_on_cpu)
        self.h5_224 = None; self.h5_ctx = None; self.merged224 = None; self.mergedCTX=None
        self.ctx_H = 1024; self.ctx_W = 1024

    def __len__(self): return len(self.records)

    def _open(self):
        if self.h5_224 is None:
            f = _h5_open_read(self.h5_224_path); self.h5_224 = f
            self.merged224 = ("images" in f and "keys" in f)
        if self.h5_ctx is None:
            f2 = _h5_open_read(self.h5_ctx_path); self.h5_ctx = f2
            self.mergedCTX = ("images" in f2 and "keys" in f2)
            if self.mergedCTX:
                ds_ctx = f2["images"]; self.ctx_H, self.ctx_W = int(ds_ctx.shape[1]), int(ds_ctx.shape[2])

    @staticmethod
    def _get224(f224, merged, k2i, key) -> np.ndarray:
        if merged:
            if k2i is None: raise KeyError(f"[224] merged H5 requires k2i: {key}")
            i = k2i.get(key); 
            if i is None: raise KeyError(f"[224] Missing key: {key}")
            arr = f224["images"][i][()]
        else:
            if key not in f224: raise KeyError(f"[224] Missing key: {key}")
            arr = f224[key][()]
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4: arr = arr[...,:3]
        if arr.dtype != np.uint8: arr = np.clip(arr,0,255).astype(np.uint8)
        if arr.shape != (224,224,3): raise ValueError(f"[224] shape bad for {key}: {arr.shape}")
        return arr

    def _read_ctx(self, f, key) -> np.ndarray:
        if self.mergedCTX:
            if self.k2iCTX is None: raise KeyError(f"[CTX] merged H5 requires k2i: {key}")
            i = self.k2iCTX.get(key); 
            if i is None: raise KeyError(f"[CTX] Missing key: {key}")
            ds = f["images"]; H,W = int(ds.shape[1]), int(ds.shape[2])
            self.ctx_H,self.ctx_W = H,W
            if self.ctx_read_mode == "strided":
                factorH = H // self.ctx_target; factorW = W // self.ctx_target
                if factorH * self.ctx_target == H and factorW * self.ctx_target == W:
                    arr = ds[i, ::factorH, ::factorW, :]
                else:
                    arr = ds[i][()]
            elif self.ctx_read_mode == "full":
                arr = ds[i][()]
            else:
                chunks = getattr(ds,"chunks",None); comp = getattr(ds,"compression",None)
                if (chunks is None) and (comp is None):
                    try:
                        factorH = H // self.ctx_target; factorW = W // self.ctx_target
                        if factorH * self.ctx_target == H and factorW * self.ctx_target == W:
                            arr = ds[i, ::factorH, ::factorW, :]
                        else:
                            arr = ds[i][()]
                    except Exception:
                        arr = ds[i][()]
                else:
                    arr = ds[i][()]
        else:
            if key not in f: raise KeyError(f"[CTX] Missing key: {key}")
            arr = f[key][()]
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4: arr = arr[...,:3]
        if arr.dtype != np.uint8: arr = np.clip(arr,0,255).astype(np.uint8)

        if self.resize_on_cpu:
            if arr.shape[0]!=self.ctx_target or arr.shape[1]!=self.ctx_target:
                t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
                t = torch.nn.functional.interpolate(t, size=(self.ctx_target,self.ctx_target), mode='bilinear', align_corners=False)
                arr = t[0].permute(1,2,0).byte().numpy()
        return arr

    def __getitem__(self, idx):
        self._open()
        r = self.records[idx]; cid = str(r["cell_id"])
        img224 = self._get224(self.h5_224, self.merged224, self.k2i224, cid)
        imgctx = self._read_ctx(self.h5_ctx, cid)

        x224 = torch.from_numpy(img224).permute(2,0,1).contiguous()
        xctx = torch.from_numpy(imgctx).permute(2,0,1).contiguous()
        y = torch.tensor(r["label"], dtype=torch.long)
        tissue = torch.tensor(r["tissue_id"], dtype=torch.long)
        return x224, xctx, y, tissue

# ===============================
# DataModule (train uses CUDA prefetch; val uses classic spawn loader)
# ===============================
class PathBDataModule(pl.LightningDataModule):
    def __init__(self, config, label_mapping, classes_to_remove):
        super().__init__()
        self.config = config
        self.label_mapping = label_mapping
        self.classes_to_remove = classes_to_remove

    @staticmethod
    def _norm_tissue(x): return str(x).strip().lower()

    def _fast_preflight(self, root: str, split: str):
        for fn in ["patches_224x224.h5", "patches_1024x1024.h5"]:
            p = os.path.join(root, split, fn)
            if not os.path.exists(p): raise FileNotFoundError(p)
            with _h5_open_read(p) as f: _ = f.id
        print(f"  preflight ok: {os.path.basename(root)}/{split}", flush=True)

    def setup(self, stage=None):
        t0 = time.perf_counter()
        print("[setup] reading annotations.json ...", flush=True)

        records = []
        for root in self.config["data_dirs"]:
            for split in ["train","test"]:
                meta = os.path.join(root, split, "annotations.json")
                if not os.path.exists(meta):
                    print(f"[WARN] {meta} not found"); continue
                df = pd.read_json(meta, orient="index").reset_index().rename(columns={'index':'cell_id'})
                df["cell_id"] = df["cell_id"].astype(str)
                if "tissue_type" not in df.columns:
                    raise KeyError(f"{meta} missing tissue_type")
                df["tissue_type"] = df["tissue_type"].apply(self._norm_tissue)
                df["split"] = split; df["data_dir"] = root
                records.extend(df.to_dict('records'))
        log_stage("annotations loaded", t0)

        self.tissue_names = sorted({r["tissue_type"] for r in records})
        self.tissue_to_idx = {n:i for i,n in enumerate(self.tissue_names)}
        self.num_tissues = len(self.tissue_names)

        filtered = [r for r in records if r.get("class_id") not in self.classes_to_remove]
        mapped = []
        for r in filtered:
            final = self.label_mapping.get(r.get("class_id"))
            if final: r["final_class"] = final; mapped.append(r)

        self.class_names = sorted({r["final_class"] for r in mapped})
        self.num_classes = len(self.class_names)
        self.class_to_idx = {n:i for i,n in enumerate(self.class_names)}

        for r in mapped:
            r["label"] = self.class_to_idx[r["final_class"]]
            r["tissue_id"] = self.tissue_to_idx[r["tissue_type"]]

        self.train_records = [r for r in mapped if r["split"]=="train"]
        self.val_records   = [r for r in mapped if r["split"]=="test"]

        labels = [r["label"] for r in self.train_records]
        self.class_weights = effective_number_weights(labels, self.num_classes, self.config["cb_beta"]) if labels else torch.ones(self.num_classes)

        t1 = time.perf_counter(); print("[setup] fast preflight ...", flush=True)
        for root in self.config["data_dirs"]:
            for split in ["train","test"]:
                meta = os.path.join(root, split, "annotations.json")
                if not os.path.exists(meta): continue
                self._fast_preflight(root, split)
        log_stage("fast preflight done", t1)

        # build H5 indices on the main process
        t2 = time.perf_counter()
        self.k2i_cache = {}
        for root in self.config["data_dirs"]:
            for split in ["train","test"]:
                p224 = os.path.join(root, split, "patches_224x224.h5")
                pCTX = os.path.join(root, split, "patches_1024x1024.h5")
                k2i224 = _load_or_build_k2i(p224)
                k2iCTX = k2i224 if _maybe_ctx_aligns_with_224(pCTX, k2i224, samples=int(self.config.get("keys_check_samples",4))) else _load_or_build_k2i(pCTX)
                self.k2i_cache[p224] = k2i224; self.k2i_cache[pCTX] = k2iCTX
        log_stage("build/load H5 key indices (main process)", t2)

        # build datasets
        t3 = time.perf_counter()
        self.train_sets, self.val_sets = [], []
        for root in self.config["data_dirs"]:
            tr = [r for r in self.train_records if r["data_dir"]==root]
            va = [r for r in self.val_records   if r["data_dir"]==root]
            if tr:
                self.train_sets.append(PathBDataset(
                    tr, os.path.join(root,"train","patches_224x224.h5"),
                    os.path.join(root,"train","patches_1024x1024.h5"),
                    self.config["ctx_img_size"],
                    k2i224=self.k2i_cache.get(os.path.join(root,"train","patches_224x224.h5")),
                    k2iCTX=self.k2i_cache.get(os.path.join(root,"train","patches_1024x1024.h5")),
                    ctx_read_mode=self.config.get("ctx_read_mode","auto"),
                    num_classes=self.num_classes,
                    resize_on_cpu=not self.config.get("gpu_resize_ctx", True),
                ))
            if va:
                self.val_sets.append(PathBDataset(
                    va, os.path.join(root,"test","patches_224x224.h5"),
                    os.path.join(root,"test","patches_1024x1024.h5"),
                    self.config["ctx_img_size"],
                    k2i224=self.k2i_cache.get(os.path.join(root,"test","patches_224x224.h5")),
                    k2iCTX=self.k2i_cache.get(os.path.join(root,"test","patches_1024x1024.h5")),
                    ctx_read_mode=self.config.get("ctx_read_mode","auto"),
                    num_classes=self.num_classes,
                    resize_on_cpu=not self.config.get("gpu_resize_ctx", True),
                ))
        log_stage("datasets built", t3)

        print(f"Classes: {self.num_classes}, Train: {len(self.train_records)}, Val: {len(self.val_records)}", flush=True)

    # ---- CUDA prefetcher (train only) ----
    class CudaPrefetcher:
        def __init__(self, loader, device="cuda"):
            self.loader = loader
            self.device = torch.device(device)
            self.stream = torch.cuda.Stream(device=self.device)
            self.next_batch = None
        def __len__(self): return len(self.loader)
        def __iter__(self):
            self.iter = iter(self.loader); self._preload(); return self
        def _to(self, x):
            if torch.is_tensor(x): return x.to(self.device, non_blocking=True)
            if isinstance(x,(list,tuple)): return type(x)(self._to(y) for y in x)
            return x
        def _preload(self):
            try: batch = next(self.iter)
            except StopIteration:
                self.next_batch = None; return
            with torch.cuda.stream(self.stream):
                self.next_batch = self._to(batch)
        def __next__(self):
            if self.next_batch is None: raise StopIteration
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            batch = self.next_batch; self._preload(); return batch

    def _make_loader(self, ds_list, shuffle, use_gpu_prefetch):
        per_device_batch = max(1, self.config["batch_size"] // max(1, self.config.get("accumulate_grad_batches",1)))
        nw = int(self.config["num_workers"])
        loader_kwargs = dict(
            batch_size=per_device_batch, shuffle=shuffle,
            num_workers=nw, pin_memory=True,
            multiprocessing_context=mp.get_context("spawn"),
        )
        if nw > 0:
            loader_kwargs["prefetch_factor"] = int(self.config.get("prefetch_factor",4))
            # persistent workers help throughput for train; disable for val for stability
            loader_kwargs["persistent_workers"] = bool(use_gpu_prefetch)
            if use_gpu_prefetch and torch.cuda.is_available():
                try: loader_kwargs["pin_memory_device"] = "cuda"
                except TypeError: pass
        loader = DataLoader(ConcatDataset(ds_list), **loader_kwargs)
        if use_gpu_prefetch and torch.cuda.is_available():
            try: loader = self.CudaPrefetcher(loader, device="cuda")
            except Exception as e: print(f"[WARN] gpu_prefetch disabled: {e}")
        return loader

    def train_dataloader(self): 
        return self._make_loader(self.train_sets, True, use_gpu_prefetch=bool(self.config.get("gpu_prefetch_train", True)))
    def val_dataloader(self):   
        return self._make_loader(self.val_sets, False, use_gpu_prefetch=bool(self.config.get("gpu_prefetch_val", False)))
    def test_dataloader(self):  return self.val_dataloader()

# ===============================
# A-head (reuses UNI features; mirrors path-local structure)
# ===============================
class FiLM(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super().__init__()
        hidden = max(64, in_dim*2)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2*feat_dim))
    def forward(self, t):
        return self.net(t).chunk(2, dim=-1)

# ===============================
# Model: UNI + A-head + DINO context + gating + A-aware weighting
# ===============================
class PathBModule(pl.LightningModule):
    def __init__(self, config, num_classes, class_names, num_tissues, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.class_names = class_names
        self.num_classes = num_classes

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)
            except AttributeError: pass

        # training/inference normalization dtype: bf16 on GPU, fp32 on CPU
        self.input_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # ---- UNI backbone (single copy) ----
        uni_cfg = {
            'img_size':224, 'patch_size':14, 'depth':24, 'num_heads':24,
            'init_values':1e-5, 'embed_dim':1536, 'mlp_ratio':2.66667*2,
            'num_classes':0, 'no_embed_class':True,
            'mlp_layer':timm.layers.SwiGLUPacked, 'act_layer':torch.nn.SiLU,
            'reg_tokens':8, 'dynamic_img_size':True
        }
        self.uni_encoder = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)
        if hasattr(self.uni_encoder, "set_grad_checkpointing"):
            self.uni_encoder.set_grad_checkpointing(self.hparams.config.get("uni_grad_ckpt", False))
        self.uni_dim = self.uni_encoder.embed_dim

        # ---- A-head branch (mirrors path-local structure) ----
        tdim = self.hparams.config["tissue_embedding_dim"]
        self.a_feat_norm = nn.LayerNorm(self.uni_dim)         # feat norm copied from path-local
        self.a_tissue_embed = nn.Embedding(num_tissues, tdim)
        self.a_tissue_film = FiLM(tdim, self.uni_dim)
        self.a_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.uni_dim, 1024), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(1024, num_classes)
        )

        # ---- B: DINO context branch ----
        cfg = AutoConfig.from_pretrained(self.hparams.config["dino_model"])
        self.dino_dim = int(cfg.hidden_size)
        self.dino_model = AutoModel.from_pretrained(
            self.hparams.config["dino_model"],
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        if self.hparams.config["freeze_dino"]:
            for p in self.dino_model.parameters(): p.requires_grad = False
            self.dino_model.eval()
        else:
            if hasattr(self.dino_model, "gradient_checkpointing_enable"):
                self.dino_model.gradient_checkpointing_enable()

        # ctx->UNI FiLM block (kept for future experiments)
        self.ctx_film_uni = FiLM(self.dino_dim, self.uni_dim)

        # ---- B projections and head ----
        self.proj_uni = nn.Sequential(nn.LayerNorm(self.uni_dim), nn.Linear(self.uni_dim, 512))
        self.proj_ctx = nn.Sequential(nn.LayerNorm(self.dino_dim), nn.Linear(self.dino_dim, 512))
        self.head_B   = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, 1024), nn.SiLU(), nn.Dropout(0.2), nn.Linear(1024, num_classes))

        # gating block (concatenate p_A(y))
        self.gate_mlp = nn.Sequential(nn.Linear(1025, 256), nn.SiLU(), nn.Linear(256, 1), nn.Sigmoid())

        # normalization stats
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

        # Loss
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.ones(num_classes))
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=self.hparams.config["label_smoothing"])

        # caches used across validation
        self.val_probs_B, self.val_labels = [], []
        self.val_probs_A, self.val_probs_fused = [], []
        self.backbone_frozen = False

        # ---- Load A-head weights (and UNI) from path-local checkpoint ----
        self._init_from_teacher_ckpt()

    # ---------- weight init ----------
    def _init_from_teacher_ckpt(self):
        ckpt = self.hparams.config.get("teacher_ckpt")
        if not ckpt or (not os.path.exists(ckpt)):
            print("[WARN] teacher_ckpt missing; A-head will be random.")
            return
        try:
            sd = torch.load(ckpt, map_location="cpu")
            state = sd.get("state_dict", sd)

            # 1) UNI backbone
            uni_keys = [k for k in state.keys() if k.startswith("uni_encoder.")]
            if uni_keys:
                missing, unexpected = self.uni_encoder.load_state_dict({k.replace("uni_encoder.",""):state[k] for k in uni_keys}, strict=False)
                print(f"[Init A] load UNI encoder: missing={len(missing)} unexpected={len(unexpected)}")

            # 2) A-head modules
            def load_sub(module, prefix_in_ckpt):
                sub = {k.replace(prefix_in_ckpt+".",""): v for k,v in state.items() if k.startswith(prefix_in_ckpt+".")}
                if sub:
                    missing, unexpected = module.load_state_dict(sub, strict=False)
                    print(f"[Init A] {prefix_in_ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
            load_sub(self.a_feat_norm, "feat_norm")
            load_sub(self.a_tissue_embed, "tissue_embedder")
            load_sub(self.a_tissue_film, "tissue_film")
            load_sub(self.a_head, "head")

            print("[Init A] A-head weights initialized.")
        except Exception as e:
            print(f"[WARN] Failed to load teacher_ckpt: {e}. A-head will be random.")

    # ---------- helpers ----------
    def _alpha(self):
        e = float(self.current_epoch)
        warm = float(self.hparams.config["ctx_alpha_warmup_epochs"])
        a0 = float(self.hparams.config["ctx_alpha_init"])
        a1 = float(self.hparams.config["ctx_alpha_final"])
        if warm <= 0 or e >= warm: return a1
        return a0 + (a1 - a0) * (e / warm)

    def _norm_inplace(self, t):
        # bf16 normalization on GPU, fp32 on CPU
        target_dtype = self.input_dtype if t.is_cuda else torch.float32
        if t.dtype != target_dtype: t = t.to(dtype=target_dtype, non_blocking=True)
        t.mul_(1.0 / 255.0)
        mean = self.pixel_mean.to(dtype=target_dtype, device=t.device, non_blocking=True)
        std  = self.pixel_std.to(dtype=target_dtype, device=t.device, non_blocking=True)
        t.sub_(mean).div_(std)
        return t

    def _extract_uni(self, x224):
        return self.uni_encoder(x224)  # [B,1536]

    def _ahead_logits(self, feat_uni, tissue):
        # A: feat_norm -> FiLM -> head
        t = self.a_tissue_embed(tissue)
        gamma, beta = self.a_tissue_film(t)
        feat = self.a_feat_norm(feat_uni)
        feat = feat * (1 + gamma) + beta
        return self.a_head(feat)  # [B,C]

    def _ctx_vec(self, xctx, training: bool):
        """Run DINO in micro-batches to keep attention memory manageable."""
        B = xctx.shape[0]
        mb = int(self.hparams.config.get("ctx_microbatch", B))
        outs = []
        for i in range(0, B, mb):
            xi = xctx[i:i+mb]
            # normalization already done; convert to model dtype to reduce memory
            xi = xi.to(dtype=self.dino_model.dtype, non_blocking=True)
            with torch.set_grad_enabled(not self.hparams.config["freeze_dino"]):
                # prefer SDPA/Fused attention when backend supports it
                out = self.dino_model(pixel_values=xi, return_dict=True)
                if self.hparams.config.get("use_pooler", True) and hasattr(out, "pooler_output") and out.pooler_output is not None:
                    vec_i = out.pooler_output
                else:
                    vec_i = out.last_hidden_state[:, 1:, :].mean(1)
            outs.append(vec_i)
        vec = torch.cat(outs, dim=0)

        if training and self.hparams.config.get("ctx_dropout", 0.0) > 0:
            if torch.rand(1, device=vec.device).item() < self.hparams.config["ctx_dropout"]:
                vec = torch.zeros_like(vec)

        # detach when DINO stays frozen
        return vec.detach() if self.hparams.config["freeze_dino"] else vec

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        x224, xctx, y, tissue = batch
        # move first, then normalize on GPU (validation dataset already CPU-resized)
        x224 = x224.to(device, non_blocking=True)
        xctx = xctx.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tissue = tissue.to(device, non_blocking=True)

        if self.hparams.config.get("gpu_resize_ctx", True):
            target = int(self.hparams.config["ctx_img_size"])
            if xctx.dim()==4 and (xctx.shape[-2]!=target or xctx.shape[-1]!=target):
                xctx = torch.nn.functional.interpolate(
                    xctx.to(dtype=torch.float32), size=(target, target),
                    mode="bilinear", align_corners=False
                ).to(dtype=x224.dtype)

        x224 = self._norm_inplace(x224).contiguous(memory_format=torch.channels_last)
        xctx = self._norm_inplace(xctx).contiguous(memory_format=torch.channels_last)
        return x224, xctx, y, tissue

    def _maybe_freeze_uni(self, freeze: bool):
        if freeze and not self.backbone_frozen:
            for p in self.uni_encoder.parameters(): p.requires_grad = False
            # also freeze A-head FiLM/Embed/Norm/Head to mimic the teacher early on
            for m in [self.a_feat_norm, self.a_tissue_embed, self.a_tissue_film, self.a_head]:
                for p in m.parameters(): p.requires_grad = False
            self.backbone_frozen = True
        if not freeze and self.backbone_frozen:
            for p in self.uni_encoder.parameters(): p.requires_grad = True
            for m in [self.a_feat_norm, self.a_tissue_embed, self.a_tissue_film, self.a_head]:
                for p in m.parameters(): p.requires_grad = True
            self.backbone_frozen = False

    def on_train_epoch_start(self):
        self._maybe_freeze_uni(self.current_epoch < self.hparams.config["freeze_uni_epochs"])

    # ---------- forward ----------
    def forward(self, x224, xctx, tissue, y=None):
        feat_uni = self._extract_uni(x224)          # [B,1536]
        z_A = self._ahead_logits(feat_uni, tissue)  # [B,C]
        p_A = torch.softmax(z_A.float(), dim=1)
        a_conf = None
        if y is not None:
            a_conf = torch.gather(p_A, 1, y.view(-1,1)).squeeze(1)  # [B]

        # B context fusion (micro-batching handled in _ctx_vec)
        ctx_vec = self._ctx_vec(xctx, self.training)                # [B,dino_dim]
        # reuse projections instead of extra FiLM blocks
        uni_res = self.proj_uni(feat_uni)                            # [B,512]
        ctx_proj = self.proj_ctx(ctx_vec)                            # [B,512]
        alpha = self._alpha()
        fused = torch.cat([uni_res, ctx_proj], dim=1)
        z_B = self.head_B(alpha * fused + (1.0 - alpha) * torch.cat([uni_res, torch.zeros_like(ctx_proj)], dim=1))

        # gating
        if a_conf is None:
            a_conf = torch.zeros((feat_uni.shape[0],), dtype=uni_res.dtype, device=uni_res.device)
        g = self.gate_mlp(torch.cat([uni_res, ctx_proj, a_conf.view(-1,1)], dim=1)).squeeze(1)  # [B]
        z_fused = z_A + g.unsqueeze(1) * z_B
        return z_A, z_B, z_fused, g, p_A, uni_res, ctx_proj

    # ---------- train / val ----------
    def training_step(self, batch, batch_idx):
        x224, xctx, y, tissue = batch
        z_A, z_B, z_fused, g, p_A, _, _ = self(x224, xctx, tissue, y=y)

        # A-aware sample weighting
        gamma = float(self.hparams.config.get("a_focus_gamma", 2.0))
        a_conf = torch.gather(p_A, 1, y.view(-1,1)).squeeze(1)  # [B]
        w = torch.clamp(1.0 - a_conf, 0, 1) ** gamma

        loss_main = torch.nn.functional.cross_entropy(z_fused, y, reduction='none', label_smoothing=self.hparams.config["label_smoothing"])
        loss_main = (loss_main * w).mean()

        # encourage B to stay calibrated
        loss_B = self.criterion(z_B, y)
        loss = loss_main + 0.3 * loss_B

        with torch.no_grad():
            acc_A = (z_A.argmax(1) == y).float().mean()
            acc_B = (z_B.argmax(1) == y).float().mean()
            acc_F = (z_fused.argmax(1) == y).float().mean()

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc_A", acc_A, on_step=True, prog_bar=False)
        self.log("train/acc_B", acc_B, on_step=True, prog_bar=True)
        self.log("train/acc_F", acc_F, on_step=True, prog_bar=False)
        return loss

    @staticmethod
    def _conf_fig(probs_np, labels_np, title):
        conf = probs_np.max(axis=1); preds = probs_np.argmax(axis=1)
        correct = (preds==labels_np).astype(np.float32)
        edges = np.linspace(0.1,1.0,10); centers=(edges[:-1]+edges[1:])/2
        cnts,accs=[],[]
        for i in range(len(edges)-1):
            lo,hi=edges[i],edges[i+1]
            m=(conf>=lo)&((conf<hi) if i<len(edges)-2 else (conf<=hi))
            cnts.append(int(m.sum())); accs.append(float(correct[m].mean()) if m.any() else 0.0)
        fig,ax1=plt.subplots(figsize=(8,4.2)); ax2=ax1.twinx()
        ax1.bar(centers,cnts,width=0.08,alpha=0.7,edgecolor='black'); ax2.plot(centers,accs,marker='o')
        ax1.set_xlabel("Confidence"); ax1.set_ylabel("Count"); ax2.set_ylabel("Accuracy"); ax1.set_title(title)
        fig.tight_layout(); return fig

    def validation_step(self, batch, batch_idx):
        x224, xctx, y, tissue = batch
        z_A, z_B, z_fused, g, p_A, _, _ = self(x224, xctx, tissue, y=y)
        probs_A = torch.softmax(z_A, dim=1)
        probs_B = torch.softmax(z_B, dim=1)
        probs_F = torch.softmax(z_fused, dim=1)
        self.val_probs_A.append(probs_A.detach().cpu())
        self.val_probs_B.append(probs_B.detach().cpu())
        self.val_probs_fused.append(probs_F.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        return None

    def on_validation_epoch_end(self):
        if not self.val_probs_B: return
        probs_A = torch.cat(self.val_probs_A).numpy()
        probs_B = torch.cat(self.val_probs_B).numpy()
        probs_F = torch.cat(self.val_probs_fused).numpy()
        labels  = torch.cat(self.val_labels).numpy()

        preds_A = probs_A.argmax(1); preds_B = probs_B.argmax(1); preds_F = probs_F.argmax(1)

        # metrics
        acc_A = accuracy_score(labels, preds_A); f1_A = f1_score(labels, preds_A, average='macro', zero_division=0)
        acc_B = accuracy_score(labels, preds_B); f1_B = f1_score(labels, preds_B, average='macro', zero_division=0)
        acc_F = accuracy_score(labels, preds_F); f1_F = f1_score(labels, preds_F, average='macro', zero_division=0)
        self.log("val/acc_Ahead", acc_A); self.log("val/f1_Ahead", f1_A)
        self.log("val/acc", acc_B, prog_bar=True); self.log("val/f1_macro", f1_B, prog_bar=True)
        self.log("val/acc_fused", acc_F, prog_bar=True); self.log("val/f1_macro_fused", f1_F, prog_bar=True)

        A_correct = (preds_A == labels); B_correct = (preds_B == labels)
        both = A_correct & B_correct
        a_only = A_correct & (~B_correct)
        b_only = (~A_correct) & B_correct
        neither = (~A_correct) & (~B_correct)
        oracle_acc = float((a_only.sum() + both.sum()) / len(labels))
        self.log("val/oracle_acc", oracle_acc, prog_bar=True)
        self.log("val/count_A_only", int(a_only.sum()))
        self.log("val/count_B_only", int(b_only.sum()))
        self.log("val/count_Both", int(both.sum()))
        self.log("val/count_Neither", int(neither.sum()))

        # visualization
        if isinstance(self.logger, pl.loggers.WandbLogger) and self.trainer.is_global_zero:
            fig, ax = plt.subplots(figsize=(6,4))
            cats = ["A_only","B_only","Both","Neither"]
            vals = [int(a_only.sum()), int(b_only.sum()), int(both.sum()), int(neither.sum())]
            ax.bar(cats, vals); ax.set_title(f"A/B Complementarity (Epoch {self.current_epoch})"); ax.set_ylabel("Count")
            for i,v in enumerate(vals): ax.text(i, v, str(v), ha='center', va='bottom')
            plt.tight_layout(); self.logger.log_image(key="val/ab_complementarity", images=[fig]); plt.close(fig)

            per_f1_B = f1_score(labels, preds_B, average=None, labels=np.arange(self.num_classes), zero_division=0)
            cm_B = confusion_matrix(labels, preds_B, normalize='true')
            fig1, ax1 = plt.subplots(figsize=(max(10,self.num_classes*0.6),4))
            sns.barplot(x=self.class_names, y=per_f1_B, ax=ax1)
            ax1.set_ylim(0,1); ax1.set_ylabel("F1"); ax1.set_title(f"Per-class F1 (B only, Epoch {self.current_epoch})")
            ax1.tick_params(axis='x', labelrotation=45); plt.setp(ax1.get_xticklabels(), ha='right')
            plt.tight_layout(); self.logger.log_image(key="val/f1_per_class_bar", images=[fig1]); plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(12,10))
            sns.heatmap(cm_B, annot=True, fmt='.2%', cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
            ax2.set_title(f"Validation Confusion Matrix (B only, Epoch {self.current_epoch}) [Path-B Stage-2]")
            ax2.tick_params(axis='x', labelrotation=45); plt.setp(ax2.get_xticklabels(), ha='right')
            plt.tight_layout(); self.logger.log_image(key="val/confusion_matrix", images=[fig2]); plt.close(fig2)

            fig3 = self._conf_fig(probs_B, labels, f"Confidence vs Accuracy/Count (B only, Epoch {self.current_epoch})")
            self.logger.log_image(key="val/confidence_accuracy_count", images=[fig3]); plt.close(fig3)

        # reset caches
        self.val_probs_A.clear(); self.val_probs_B.clear(); self.val_probs_fused.clear(); self.val_labels.clear()

    # ---------- optim ----------
    def configure_optimizers(self):
        wd = self.hparams.config["weight_decay"]
        lr_uni = self.hparams.config["lr_uni"]
        lr_head = self.hparams.config["lr_head"]
        gamma = self.hparams.config["lrd_gamma"]
        param_groups=[]
        from torch.nn import Parameter
        def add(obj, lr):
            if obj is None: return
            if isinstance(obj, Parameter):
                if obj.requires_grad: param_groups.append({"params":[obj],"lr":lr,"weight_decay":wd})
            else:
                for p in obj.parameters(recurse=True):
                    if p.requires_grad: param_groups.append({"params":[p],"lr":lr,"weight_decay":wd})
        # UNI LLRD
        vit_blocks = list(getattr(self.uni_encoder,"blocks",[]))
        n=len(vit_blocks); low_lr = lr_uni*(gamma**n)
        add(getattr(self.uni_encoder,"patch_embed",None), low_lr)
        add(getattr(self.uni_encoder,"pos_embed", None), low_lr)
        add(getattr(self.uni_encoder,"pos_drop", None), low_lr)
        add(getattr(self.uni_encoder,"cls_token", None), low_lr)
        add(getattr(self.uni_encoder,"reg_token", None), low_lr)
        for i,blk in enumerate(vit_blocks): add(blk, lr_uni*(gamma**(n-i-1)))
        add(getattr(self.uni_encoder,"norm",None), lr_uni)

        # A-head, B-head, projections, gate, ctx-FiLM
        add(self.a_feat_norm, lr_head); add(self.a_tissue_embed, lr_head); add(self.a_tissue_film, lr_head); add(self.a_head, lr_head)
        add(self.ctx_film_uni, lr_head); add(self.proj_uni, lr_head); add(self.proj_ctx, lr_head); add(self.head_B, lr_head); add(self.gate_mlp, lr_head)

        # DINO
        if not self.hparams.config["freeze_dino"]:
            add(self.dino_model, self.hparams.config["lr_dino"])

        try:
            optimizer = torch.optim.AdamW(param_groups, weight_decay=wd, fused=bool(self.hparams.config.get("optimizer_fused", True)))
        except TypeError:
            optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, (step+1)/max(1, self.hparams.config["warmup_steps"]))
        )
        return {"optimizer":optimizer, "lr_scheduler":{"scheduler":scheduler, "interval":"step"}}

# ===============================
# Inference dump (B-only; keeps released structure)
# ===============================
@torch.no_grad()
def run_inference_and_dump(model_ckpt_path, config, dm: PathBDataModule, class_names: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PathBModule.load_from_checkpoint(
        checkpoint_path=model_ckpt_path,
        config=config, num_classes=len(class_names), class_names=class_names,
        num_tissues=dm.num_tissues, class_weights=dm.class_weights,
        map_location=device, strict=False
    ).to(device).eval()

    with open(os.path.join(out_dir,"class_names.json"),"w") as f: json.dump(class_names,f,indent=2)
    with open(os.path.join(out_dir,"class_to_idx.json"),"w") as f: json.dump({n:i for i,n in enumerate(class_names)},f,indent=2)
    with open(os.path.join(out_dir,"tissue_names.json"),"w") as f: json.dump(dm.tissue_names,f,indent=2)
    with open(os.path.join(out_dir,"tissue_to_idx.json"),"w") as f: json.dump(dm.tissue_to_idx,f,indent=2)
    with open(os.path.join(out_dir,"config.json"),"w") as f: json.dump(config,f,indent=2)

    loader = dm.val_dataloader()
    all_logits, all_probs, all_labels, all_fused = [], [], [], []
    for batch in loader:
        x224,xctx,y,tissue = batch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            _, z_B, _, _, _, uni_res, ctx_proj = model(x224, xctx, tissue, y=None)
            probs = torch.softmax(z_B, dim=1)
            fused = torch.cat([uni_res, ctx_proj], dim=1)
        all_logits.append(z_B.float().cpu()); all_probs.append(probs.float().cpu()); all_labels.append(y.cpu()); all_fused.append(fused.float().cpu())

    logits_np = torch.cat(all_logits).numpy()
    probs_np  = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    fused_np  = torch.cat(all_fused).numpy()
    preds_np  = probs_np.argmax(axis=1)

    metrics = {"acc": float(accuracy_score(labels_np, preds_np)),
               "f1_macro": float(f1_score(labels_np, preds_np, average='macro', zero_division=0))}
    try:
        metrics["auroc_macro_ovr"] = float(roc_auc_score(labels_np, probs_np, multi_class='ovr', average='macro'))
    except Exception:
        metrics["auroc_macro_ovr"] = None

    np.save(os.path.join(out_dir,"embeddings.npy"), fused_np)
    np.save(os.path.join(out_dir,"probe.npy"),      fused_np)
    np.save(os.path.join(out_dir,"logits.npy"),     logits_np)
    np.save(os.path.join(out_dir,"probs.npy"),      probs_np)
    np.save(os.path.join(out_dir,"labels.npy"),     labels_np)
    np.save(os.path.join(out_dir,"preds.npy"),      preds_np)
    with open(os.path.join(out_dir,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    print(f"[Path-B Stage-2] Inference dumps saved to: {out_dir} | {metrics}")

# ===============================
# Entry
# ===============================
def main():
    config = prepare_config()

    seed_everything(config["seed"], workers=True)
    torch.set_float32_matmul_precision('medium')
    assert_cuda_arch_compat()

    if torch.cuda.is_available():
        accelerator, devices, precision = 'gpu', 1, 'bf16-mixed'
        print(f"[INFO] Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        print(f"[INFO] Training on cuda:0 -> {torch.cuda.get_device_name(0)}", flush=True)
    else:
        accelerator, devices, precision = 'cpu', 1, '32'
        print("[INFO] CUDA not available, using CPU.", flush=True)

    dm = PathBDataModule(config, NEW_LABEL_MAPPING, CLASSES_TO_REMOVE)
    dm.setup()

    model = PathBModule(
        config=config, num_classes=dm.num_classes, class_names=dm.class_names,
        num_tissues=dm.num_tissues, class_weights=dm.class_weights
    )

    # optional torch.compile (off by default)
    if config.get("torch_compile", False) and hasattr(torch, "compile"):
        try:
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="default")
            print("[INFO] torch.compile enabled (auto-fallback).")
        except Exception as e:
            print(f"[WARN] torch.compile disabled: {e}")

    # W&B
    wandb_logger = None
    try:
        import pytorch_lightning.loggers as pl_loggers
        wandb_logger = pl_loggers.WandbLogger(
            name=config["experiment_name"], project=config["project_name"],
            entity=config["entity"], config=config, resume='allow'
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
        accelerator=accelerator, devices=devices, precision=precision, strategy='auto',
        max_epochs=config["max_epochs"], val_check_interval=config["val_check_interval"],
        gradient_clip_val=config.get("gradient_clip_val", 0.0),
        callbacks=[TQDMProgressBar(refresh_rate=10), early_cb, ckpt_cb],
        logger=wandb_logger, log_every_n_steps=10,
        accumulate_grad_batches=config["accumulate_grad_batches"], enable_checkpointing=True,
        # skip sanity check to avoid very slow first iteration on DINO-L/16
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dm)

    best = ckpt_cb.best_model_path
    if best and os.path.exists(best):
        print(f"[Path-B Stage-2] Best checkpoint: {best}")
        run_inference_and_dump(best, config, dm, dm.class_names, config["inference_out_dir"])
    else:
        print("[Path-B Stage-2] No best checkpoint found; skip inference dump.")

if __name__ == "__main__":
    main()
