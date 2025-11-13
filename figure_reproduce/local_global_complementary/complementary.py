#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A/B complementarity (no fusion) evaluation script.
- Loads Path-A / Path-B checkpoints (Lightning-style state_dict exports)
- Reads validation splits (split='test') annotations + H5 patches at 224 and 1024
- Runs inference on the shared cell_id intersection to count A-only / B-only / Both / Neither
- Prints a summary to stdout and saves a CSV under the chosen output directory
"""

import os, json, argparse, warnings, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # allow safe concurrent reads

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from Nuclass.utils.torchvision_compat import ensure_torchvision_nms

ensure_torchvision_nms()
import timm
from transformers import AutoModel, AutoConfig, logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*No device id is provided.*", category=UserWarning)

# -----------------------------
# Default paths (override via CLI)
# -----------------------------
DEFAULT_CKPT_A = Path("checkpoints") / "path_local_release" / "best.ckpt"
DEFAULT_CKPT_B = Path("checkpoints") / "path_global_release" / "best.ckpt"
DEFAULT_RESULTS_DIR = Path("results") / "figure_reproduce" / "local_global_complementary"
CSV_NAME = "ab_complementarity_no_fusion.csv"

try:
    from Nuclass.train.label_mappings import NEW_LABEL_MAPPING, CLASSES_TO_REMOVE
    from Nuclass.train.dataset_defaults import DEFAULT_DATASET_FOLDERS
    from Nuclass.train.config_utils import resolve_data_dirs
except ModuleNotFoundError:
    from label_mappings import NEW_LABEL_MAPPING, CLASSES_TO_REMOVE  # type: ignore
    from dataset_defaults import DEFAULT_DATASET_FOLDERS  # type: ignore
    from config_utils import resolve_data_dirs  # type: ignore


def _resolve_dirs(explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return [os.path.abspath(p) for p in explicit]
    return resolve_data_dirs(DEFAULT_DATASET_FOLDERS)

# -----------------------------
# Helpers
# -----------------------------
def assert_cuda_arch_compat():
    if not torch.cuda.is_available(): return
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 9:  # H100/H200 = sm_90
        ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if ver < (2, 1):
            raise SystemExit(f"Requires torch>=2.1 (CUDA12) to run on sm_90 GPUs, found {torch.__version__}")

def read_annotations(data_dirs: List[str], split: str) -> List[Dict]:
    records = []
    for root in data_dirs:
        meta = os.path.join(root, split, "annotations.json")
        if not os.path.exists(meta):
            print(f"[WARN] missing {meta}, skipping", flush=True)
            continue
        with open(meta, "r") as f:
            raw = json.load(f)  # expected structure: {cell_id: {...}}
        for cid, info in raw.items():
            r = dict(info)
            r["cell_id"] = str(cid)
            r["data_dir"] = root
            r["split"] = split
            if "tissue_type" not in r:
                raise KeyError(f"{meta} missing 'tissue_type'")
            r["tissue_type"] = str(r["tissue_type"]).strip().lower()
            # Label field could be named class_id / label / cell_type; prefer class_id
            cls = r.get("class_id", r.get("label", r.get("cell_type")))
            r["class_id"] = cls
            records.append(r)
    return records

def map_and_filter(records: List[Dict]) -> List[Dict]:
    out = []
    for r in records:
        cid = r.get("class_id")
        if cid in CLASSES_TO_REMOVE:
            continue
        final = NEW_LABEL_MAPPING.get(cid)
        if final:
            rr = dict(r)
            rr["final_class"] = final
            out.append(rr)
    return out

def build_vocab(records_all: List[Dict]) -> Tuple[List[str], Dict[str,int], List[str], Dict[str,int]]:
    tissue_names = sorted({r["tissue_type"] for r in records_all})
    class_names  = sorted({r["final_class"] for r in records_all})
    tissue_to_idx = {n:i for i,n in enumerate(tissue_names)}
    class_to_idx  = {n:i for i,n in enumerate(class_names)}
    return tissue_names, tissue_to_idx, class_names, class_to_idx

def open_h5(path: str) -> h5py.File:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return h5py.File(path, "r")

def build_k2i(f: h5py.File) -> Optional[Dict[str,int]]:
    if "keys" in f:
        try:
            keys = f["keys"].asstr()[()]
        except Exception:
            raw = f["keys"][()]
            keys = [k.decode() if isinstance(k,(bytes,np.bytes_)) else str(k) for k in raw]
        return {str(k): int(i) for i,k in enumerate(keys)}
    return None

def get_img_from_h5(f: h5py.File, cid: str, k2i: Optional[Dict[str,int]]) -> np.ndarray:
    if "images" in f and k2i is not None:
        idx = k2i.get(cid)
        if idx is None:
            raise KeyError(f"Merged H5 missing key: {cid}")
        arr = f["images"][idx][()]
    else:
        if cid not in f:
            raise KeyError(f"H5 missing key: {cid}")
        arr = f[cid][()]
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def to_norm_tensor_u8(chw_u8: torch.Tensor, device, dtype=torch.bfloat16) -> torch.Tensor:
    # Accepts uint8 tensors shaped [B,3,H,W] or [3,H,W]
    if chw_u8.dim() == 3:
        chw_u8 = chw_u8.unsqueeze(0)
    x = chw_u8.to(device=device, dtype=torch.float32) / 255.0
    mean = IMNET_MEAN.to(device=device, dtype=torch.float32)
    std  = IMNET_STD.to(device=device, dtype=torch.float32)
    x = (x - mean) / std
    return x.to(dtype=dtype)

# -----------------------------
# Minimal model definitions
# -----------------------------
class FiLM(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int):
        super().__init__()
        hidden = max(64, in_dim*2)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2*feat_dim))
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.net(t)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta

class AOnlyPredictor(nn.Module):
    def __init__(self, num_classes: int, num_tissues: int, tissue_dim: int = 64):
        super().__init__()
        uni_cfg = {
            'img_size':224, 'patch_size':14, 'depth':24, 'num_heads':24,
            'init_values':1e-5, 'embed_dim':1536, 'mlp_ratio':2.66667*2,
            'num_classes':0, 'no_embed_class':True,
            'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
            'reg_tokens':8, 'dynamic_img_size':True
        }
        self.uni_encoder = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)
        self.uni_dim = self.uni_encoder.embed_dim
        self.feat_norm = nn.LayerNorm(self.uni_dim)
        self.tissue_embedder = nn.Embedding(num_tissues, tissue_dim)
        self.tissue_film = FiLM(tissue_dim, self.uni_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.uni_dim, 1024), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(1024, num_classes)
        )

    @torch.no_grad()
    def forward(self, x224_bchw, tissue_b):
        feat = self.uni_encoder(x224_bchw)
        feat = self.feat_norm(feat)
        t = self.tissue_embedder(tissue_b)
        gamma, beta = self.tissue_film(t)
        feat = feat * (1 + gamma) + beta
        return self.head(feat)

    def load_ckpt_state(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        state = sd.get("state_dict", sd)
        keep = {k: v for k, v in state.items() if k.startswith(("uni_encoder.", "feat_norm.", "tissue_embedder.", "tissue_film.", "head."))}
        missing, unexpected = self.load_state_dict(keep, strict=False)
        print(f"[A] load ckpt: missing={len(missing)} unexpected={len(unexpected)}")

class BOnlyPredictor(nn.Module):
    def __init__(self, num_classes: int, dino_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
                 alpha: float = 1.0, ctx_microbatch: int = 16):
        super().__init__()
        self.alpha = float(alpha)
        self.ctx_microbatch = int(ctx_microbatch)

        uni_cfg = {
            'img_size':224, 'patch_size':14, 'depth':24, 'num_heads':24,
            'init_values':1e-5, 'embed_dim':1536, 'mlp_ratio':2.66667*2,
            'num_classes':0, 'no_embed_class':True,
            'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
            'reg_tokens':8, 'dynamic_img_size':True
        }
        self.uni_encoder = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)
        self.uni_dim = self.uni_encoder.embed_dim

        cfg = AutoConfig.from_pretrained(dino_model)
        self.dino_dim = int(cfg.hidden_size)
        self.dino_model = AutoModel.from_pretrained(dino_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

        self.proj_uni = nn.Sequential(nn.LayerNorm(self.uni_dim), nn.Linear(self.uni_dim, 512))
        self.proj_ctx = nn.Sequential(nn.LayerNorm(self.dino_dim), nn.Linear(self.dino_dim, 512))
        self.head_B   = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, 1024), nn.SiLU(), nn.Dropout(0.2), nn.Linear(1024, num_classes))

    @torch.no_grad()
    def forward_b(self, x224_bchw, xctx_bchw):
        # UNI
        feat_uni = self.uni_encoder(x224_bchw)           # [B,1536]
        uni_res = self.proj_uni(feat_uni)                # [B,512]

        # DINO forward in micro-batches
        outs = []
        B = xctx_bchw.shape[0]
        for i in range(0, B, self.ctx_microbatch):
            xi = xctx_bchw[i:i+self.ctx_microbatch]
            out = self.dino_model(pixel_values=xi, return_dict=True)
            vec = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None \
                  else out.last_hidden_state[:, 1:, :].mean(1)
            outs.append(vec)
        ctx_vec = torch.cat(outs, dim=0)                 # [B,dino_dim]
        ctx_proj = self.proj_ctx(ctx_vec)                # [B,512]

        fused = torch.cat([uni_res, ctx_proj], dim=1)    # [B,1024]
        zB = self.head_B(self.alpha * fused + (1.0 - self.alpha) *
                         torch.cat([uni_res, torch.zeros_like(ctx_proj)], dim=1))
        return zB

    def load_ckpt_state(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        state = sd.get("state_dict", sd)
        # Load B-branch weights only; fall back to HF weights if DINO blocks missing
        keep = {k: v for k, v in state.items() if k.startswith(("uni_encoder.", "proj_uni.", "proj_ctx.", "head_B.", "dino_model."))}
        missing, unexpected = self.load_state_dict(keep, strict=False)
        print(f"[B] load ckpt: missing={len(missing)} unexpected={len(unexpected)}")

# -----------------------------
# Validation datasets (per-root H5 readers)
# -----------------------------
class AValDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        """
        samples: [{cell_id, data_dir, tissue_type, final_class}]
        """
        super().__init__()
        self.samples = samples
        self._h5_cache: Dict[str, Dict] = {}  # root -> {"f224":h5, "k2i":dict}
    def __len__(self): return len(self.samples)

    def _ensure_open(self, root: str):
        if root in self._h5_cache: return
        p224 = os.path.join(root, "test", "patches_224x224.h5")
        f224 = open_h5(p224); k2i = build_k2i(f224)
        self._h5_cache[root] = {"f224": f224, "k2i": k2i}

    def __getitem__(self, idx):
        r = self.samples[idx]
        root, cid = r["data_dir"], r["cell_id"]
        self._ensure_open(root)
        f224, k2i = self._h5_cache[root]["f224"], self._h5_cache[root]["k2i"]
        img = get_img_from_h5(f224, cid, k2i)  # HWC uint8
        if img.shape != (224,224,3):
            raise ValueError(f"[A] {root} {cid} expected 224x224x3, got {img.shape}")
        x224 = torch.from_numpy(img).permute(2,0,1).contiguous()  # CHW uint8
        return x224, r["tissue_type"], r["final_class"], cid

class BValDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        super().__init__()
        self.samples = samples
        self._h5_cache: Dict[str, Dict] = {}  # root -> {"f224":h5,"k2i_224":dict,"f1024":h5,"k2i_1024":dict}
    def __len__(self): return len(self.samples)

    def _ensure_open(self, root: str):
        if root in self._h5_cache: return
        p224 = os.path.join(root, "test", "patches_224x224.h5")
        p1024= os.path.join(root, "test", "patches_1024x1024.h5")
        f224 = open_h5(p224); k2i_224 = build_k2i(f224)
        f1024= open_h5(p1024); k2i_1024= build_k2i(f1024)
        self._h5_cache[root] = {"f224":f224, "k2i_224":k2i_224, "f1024":f1024, "k2i_1024":k2i_1024}

    def __getitem__(self, idx):
        r = self.samples[idx]
        root, cid = r["data_dir"], r["cell_id"]
        self._ensure_open(root)
        cache = self._h5_cache[root]
        img224 = get_img_from_h5(cache["f224"], cid, cache["k2i_224"])
        if img224.shape != (224,224,3):
            raise ValueError(f"[B-224] {root} {cid} expected 224x224x3, got {img224.shape}")
        x224 = torch.from_numpy(img224).permute(2,0,1).contiguous()

        img1024 = get_img_from_h5(cache["f1024"], cid, cache["k2i_1024"])
        if img1024.shape[:2] != (1024,1024):
            # Some datasets provide 512 directly; check both possibilities
            if img1024.shape[:2] != (512,512):
                raise ValueError(f"[B-ctx] {root} {cid} expected 1024x1024x3 or 512x512x3, got {img1024.shape}")
        xctx = torch.from_numpy(img1024).permute(2,0,1).contiguous()  # CHW uint8 (1024 or 512)
        return x224, xctx, r["final_class"], cid

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="A/B Complementarity (no fusion) from ckpt")
    parser.add_argument("--ckptA", default=str(DEFAULT_CKPT_A), type=str)
    parser.add_argument("--ckptB", default=str(DEFAULT_CKPT_B), type=str)
    parser.add_argument("--alphaB", default=1.0, type=float, help="Alpha weighting for the B-only branch (default 1.0)")
    parser.add_argument("--ctx-microbatch", default=16, type=int, help="DINO forward micro-batch size")
    parser.add_argument("--bsA", default=512, type=int)
    parser.add_argument("--bsB", default=64, type=int)
    parser.add_argument("--num-workers", default=0, type=int, help="Number of dataloader workers (default 0 for simpler H5 handling)")
    parser.add_argument("--data-dirs-A", nargs="+", default=None, help="Optional override for Path-A datasets")
    parser.add_argument("--data-dirs-B", nargs="+", default=None, help="Optional override for Path-B datasets")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for CSV/log outputs")
    args = parser.parse_args()

    assert_cuda_arch_compat()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("medium")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    data_dirs_a = _resolve_dirs(args.data_dirs_A)
    data_dirs_b = _resolve_dirs(args.data_dirs_B)
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect validation records (A/B both use split='test') ----
    recsA_all = map_and_filter(read_annotations(data_dirs_a, "test"))
    recsB_all = map_and_filter(read_annotations(data_dirs_b, "test"))

    # Build vocab (A needs tissues/classes for the FiLM head; B logging only)
    tissuesA, tissue_to_idxA, classesA, class_to_idxA = build_vocab(recsA_all)
    tissuesB, tissue_to_idxB, classesB, class_to_idxB = build_vocab(recsB_all)  # logging only

    # Index cell_id -> record
    idxA_by_cid = {r["cell_id"]: r for r in recsA_all}
    idxB_by_cid = {r["cell_id"]: r for r in recsB_all}
    inter_ids = sorted(set(idxA_by_cid.keys()) & set(idxB_by_cid.keys()))
    if not inter_ids:
        print("[ERROR] No cell_id intersection between A/B validation sets; verify matching data directories.")
        sys.exit(1)

    # Keep samples with consistent GT annotations
    keep_ids = []
    bad = 0
    for cid in inter_ids:
        if idxA_by_cid[cid]["final_class"] == idxB_by_cid[cid]["final_class"]:
            keep_ids.append(cid)
        else:
            bad += 1
    if bad > 0:
        print(f"[WARN] removed {bad} samples due to A/B GT mismatches")
    if len(keep_ids) == 0:
        print("[ERROR] No evaluable samples remain after cleaning")
        sys.exit(1)

    # Assemble evaluation sample lists
    samplesA = [{
        "cell_id": cid,
        "data_dir": idxA_by_cid[cid]["data_dir"],
        "tissue_type": idxA_by_cid[cid]["tissue_type"],
        "final_class": idxA_by_cid[cid]["final_class"],
    } for cid in keep_ids]
    samplesB = [{
        "cell_id": cid,
        "data_dir": idxB_by_cid[cid]["data_dir"],
        "final_class": idxB_by_cid[cid]["final_class"],
    } for cid in keep_ids]

    # ---- DataLoader ----
    dlA = DataLoader(AValDataset(samplesA), batch_size=args.bsA, shuffle=False,
                     num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    dlB = DataLoader(BValDataset(samplesB), batch_size=args.bsB, shuffle=False,
                     num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # ---- Build and load models ----
    modelA = AOnlyPredictor(num_classes=len(classesA), num_tissues=len(tissuesA)).to(device).eval()
    modelA.load_ckpt_state(args.ckptA)

    modelB = BOnlyPredictor(num_classes=len(classesB), alpha=args.alphaB,
                            ctx_microbatch=args.ctx_microbatch).to(device).eval()
    modelB.load_ckpt_state(args.ckptB)

    # ---- Inference: Path-A ----
    predA_by_cid: Dict[str, str] = {}
    gt_by_cid: Dict[str, str] = {}
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device if device=="cuda" else "cpu", dtype=torch.bfloat16 if device=="cuda" else torch.float32):
        for x224_u8, tissue_name, gt_name, cid in dlA:
            # Normalize
            x224 = to_norm_tensor_u8(x224_u8, device=device, dtype=torch.bfloat16 if device=="cuda" else torch.float32)
            tissue_idx = torch.tensor([tissue_to_idxA[str(t)] for t in tissue_name], dtype=torch.long, device=device)
            logits = modelA(x224, tissue_idx)
            preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            for c, pi, gtn in zip(cid, preds, gt_name):
                predA_by_cid[str(c)] = classesA[pi]
                gt_by_cid[str(c)] = str(gtn)
    print(f"[A] inference done, samples={len(predA_by_cid)}, time={time.perf_counter()-t0:.1f}s")

    # ---- Inference: Path-B (ctx resized to 512) ----
    predB_by_cid: Dict[str, str] = {}
    t1 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device if device=="cuda" else "cpu", dtype=torch.bfloat16 if device=="cuda" else torch.float32):
        for x224_u8, xctx_u8, gt_name, cid in dlB:
            x224 = to_norm_tensor_u8(x224_u8, device=device, dtype=torch.bfloat16 if device=="cuda" else torch.float32)
            # Resize ctx to 512x512 on the fly (GPU interpolation)
            xctx = xctx_u8.to(device=device, dtype=torch.float32) / 255.0
            if xctx.shape[-2:] != (512, 512):
                xctx = F.interpolate(xctx, size=(512,512), mode="bilinear", align_corners=False)
            # Normalize
            mean = IMNET_MEAN.to(device=device, dtype=torch.float32)
            std  = IMNET_STD.to(device=device, dtype=torch.float32)
            xctx = ((xctx - mean) / std).to(dtype=torch.bfloat16 if device=="cuda" else torch.float32)

            logitsB = modelB.forward_b(x224, xctx)
            predsB = logitsB.argmax(dim=1).detach().cpu().numpy().tolist()
            for c, pi in zip(cid, predsB):
                predB_by_cid[str(c)] = classesB[pi]
    print(f"[B] inference done, samples={len(predB_by_cid)}, time={time.perf_counter()-t1:.1f}s")

    # ---- Complementarity stats ----
    ids = [cid for cid in keep_ids if cid in predA_by_cid and cid in predB_by_cid]
    if not ids:
        print("[ERROR] Empty sample intersection after predictions")
        sys.exit(1)

    gt_names = np.array([gt_by_cid[cid] for cid in ids], dtype=object)
    predA    = np.array([predA_by_cid[cid] for cid in ids], dtype=object)
    predB    = np.array([predB_by_cid[cid] for cid in ids], dtype=object)

    accA = float((predA == gt_names).mean())
    accB = float((predB == gt_names).mean())
    oracle = float(((predA == gt_names) | (predB == gt_names)).mean())

    classes_in_data = sorted(list(set(gt_names.tolist())))
    rows = []
    total_a_only = total_b_only = total_both = total_neither = 0
    for cls in classes_in_data:
        m = (gt_names == cls)
        n = int(m.sum())
        a_correct = (predA == cls) & m
        b_correct = (predB == cls) & m
        a_only = int((a_correct & ~b_correct).sum())
        b_only = int((~a_correct & b_correct).sum())
        both   = int((a_correct & b_correct).sum())
        neither= int((~a_correct & ~b_correct).sum())
        total_a_only += a_only; total_b_only += b_only; total_both += both; total_neither += neither
        disagree = a_only + b_only
        if disagree > 0:
            a_share = a_only / disagree
            dom = "A" if a_share > 0.5 else ("B" if a_share < 0.5 else "~50/50")
        else:
            a_share = None; dom = "-"
        rows.append({
            "class": cls, "N": n,
            "A_only": a_only, "B_only": b_only, "Both": both, "Neither": neither,
            "A_only_pct": (a_only / n) if n>0 else 0.0,
            "B_only_pct": (b_only / n) if n>0 else 0.0,
            "A_share_disagree": a_share, "dominant": dom,
        })

    # Summarize which branch dominates when disagreement exists
    with_disagree = [r for r in rows if (r["A_only"] + r["B_only"]) > 0]
    b_dom = sum(1 for r in with_disagree if r["dominant"] == "B")
    a_dom = sum(1 for r in with_disagree if r["dominant"] == "A")
    tie   = sum(1 for r in with_disagree if r["dominant"] == "~50/50")

    # ---- Logging ----
    def fmt_pct(x, d): return ("-" if d<=0 else f"{100.0*x/d:5.1f}%")
    print("="*96)
    print("A/B Complementarity (no fusion) - from ckpt")
    print(f"A ckpt: {args.ckptA}")
    print(f"B ckpt: {args.ckptB}")
    print(f"Aligned samples: {len(ids)}")
    print(f"Acc(A)={accA:.4f} | Acc(B)={accB:.4f} | Oracle(A or B)={oracle:.4f} | Gain={oracle - max(accA, accB):+.4f}")
    print(f"Counts: A_only={total_a_only} | B_only={total_b_only} | Both={total_both} | Neither={total_neither}")
    print(f"Classes with disagreement: {len(with_disagree)}  ->  B dominant: {b_dom} / A dominant: {a_dom} / ~50-50: {tie}")
    print("-"*96)
    header = ["Class","N","A_only","B_only","Both","Neither","A_only%","B_only%","A-share(disagree)","Dominant"]
    colw = [max(len(h), 10) for h in header]
    def pr_row(vals):
        s = ""
        for i,v in enumerate(vals):
            s += str(v).ljust(colw[i]+2)
        print(s)
    pr_row(header); print("-"*96)
    rows_sorted = sorted(rows, key=lambda r: r["class"])
    for r in rows_sorted:
        a_share = "-" if r["A_share_disagree"] is None else f"{100.0*r['A_share_disagree']:5.1f}%"
        pr_row([
            r["class"], r["N"], r["A_only"], r["B_only"], r["Both"], r["Neither"],
            fmt_pct(r["A_only"], r["N"]), fmt_pct(r["B_only"], r["N"]), a_share, r["dominant"]
        ])
    print("="*96)

    # ---- Save CSV ----
    out_csv = os.path.join(results_dir, CSV_NAME)
    with open(out_csv, "w") as f:
        f.write(",".join(["class","N","A_only","B_only","Both","Neither",
                          "A_only_pct","B_only_pct","A_share_disagree","dominant"]) + "\n")
        for r in rows_sorted:
            f.write("{},{},{},{},{},{},{:.6f},{:.6f},{}{},{}\n".format(
                r["class"], r["N"], r["A_only"], r["B_only"], r["Both"], r["Neither"],
                r["A_only_pct"], r["B_only_pct"],
                "" if r["A_share_disagree"] is None else f"{r['A_share_disagree']:.6f}",
                "" if r["A_share_disagree"] is not None else "",
                r["dominant"]
            ).replace("None",""))
    print(f"[SAVE] CSV -> {out_csv}")

if __name__ == "__main__":
    main()
