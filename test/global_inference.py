# -*- coding: utf-8 -*-
"""
Path-B inference (release)
- Mirrors the training module sub-structure:
  uni_encoder / a_feat_norm / a_tissue_embed / a_tissue_film(net) / a_head
  dino_model / proj_uni / proj_ctx / head_B / gate_mlp
- GPU handling: always uses device 0 (honors CUDA_VISIBLE_DEVICES externally)
- H5 reader: streams 224 + 1024 tiles together; 1024 loader supports stride slicing (::factor)
- DINO forward runs in micro-batches to avoid memory spikes
- Holdout label alignment:
  * Xenium: strict one-to-one (drop {'SMC/Pericyte','Myeloid'}, normalize synonyms)
  * Lizard: collapse into three groups (plasma->Lymphocyte; drop neutrophil/eosinophil)
- Outputs match A/PLIP/LOKI conventions:
  embeddings.npy (= fused 1024D), probe.npy (= embeddings), probs.npy (eval classes),
  labels.npy, preds.npy, metrics.json
"""

import os, json, time, warnings, argparse, h5py, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")       # silence TF protobuf import
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")     # hide TF INFO logs
warnings.filterwarnings("ignore", message=r".*No device id is provided.*")
warnings.filterwarnings("ignore", message=r"Protobuf gencode version .*", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from Nuclass.utils.torchvision_compat import ensure_torchvision_nms
from Nuclass.utils.backbone_paths import resolve_dino_model_path

ensure_torchvision_nms()
import timm
from transformers import AutoModel, AutoConfig, logging as hf_logging
hf_logging.set_verbosity_error()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "path_global_release" / "best.ckpt"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "test" / "path_b_inference"
DEFAULT_HOLDOUTS_FILE = Path(__file__).resolve().parent / "configs" / "holdouts_template.json"


def load_holdouts_config(path: Path) -> Dict[str, Dict[str, str]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if isinstance(cfg, list):
        cfg = {item.pop("name"): item for item in cfg}
    if not isinstance(cfg, dict):
        raise ValueError("Holdouts config must be an object or list of {name,...}.")
    normalized = {}
    for name, info in cfg.items():
        if not isinstance(info, dict):
            raise ValueError(f"Holdout '{name}' must map to an object.")
        missing = {"root", "mode", "tissue"} - info.keys()
        if missing:
            raise ValueError(f"Holdout '{name}' missing fields: {missing}")
        normalized[name] = {
            "root": os.path.expanduser(str(info["root"])),
            "mode": str(info["mode"]),
            "tissue": str(info["tissue"])
        }
    return normalized


TISSUE_TO_IDX = {"breast":0,"colon":1,"heart":2,"kidney":3,"liver":4,"lung":5,"ovary":6,"pancreas":7}

# ============== Label mappings (must mirror training) ==============
# Synonyms (Xenium)
SYN_ALIAS = {"T_cells":"T_Cell","Fibroblasts":"Fibroblast","Endothelial Cell":"Endothelial","Macrophages":"Macrophage","Tumor_epithelial":"Epithelial_Malignant"}
EXACT_REMOVE = {"SMC/Pericyte","Myeloid"}
EXACT_HOLDOUT_TO_TRAIN = {
    "Epithelial_Malignant":"Epithelial_Malignant","Fibroblast":"Fibroblast","Macrophage":"Macrophage","Endothelial":"Endothelial",
    "Ductal":"Ductal","Acinar":"Acinar","Endocrine":"Endocrine","Monocyte":"Monocyte","T_Cell":"T_Cell","B_Cell":"B_Cell","NK_Cell":"NK_Cell",
}

# Lizard grouping (plasma->Lymphocyte; drop neutrophil/eosinophil)
LIZARD_RAW_TO_GROUP = {"epithelial":"Epithelial","connective":"Connective","lymphocyte":"Lymphocyte","plasma":"Lymphocyte"}
LIZARD_GROUP_TO_TRAIN = {
    "Epithelial": ["Epithelial_Malignant","Ductal","Acinar","Epithelial_Airway","Epithelial_Alveolar"],
    "Connective": ["Fibroblast","Tumor_Associated_Fibroblast","Smooth_Muscle"],
    "Lymphocyte": ["T_Cell","B_Cell","NK_Cell","Plasma_Cell","Macrophage","Monocyte"],
}

# ============== Constants ==============
IM_MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
IM_STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

def norm_img_uint8_to_float(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device=device, dtype=torch.float32).div_(255.0)
    return x.sub_(IM_MEAN.to(device)).div_(IM_STD.to(device))

# ============== H5 batched reader (224 + 1024) ==============
class H5DualReader:
    def __init__(self, p224: str, p1024: str, ctx_size: int):
        self.p224, self.p1024, self.ctx = p224, p1024, int(ctx_size)
        for p in [p224,p1024]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        self.f224=self.f1024=None; self.m224=self.m1024=False
        self.k2i224=self.k2i1024=None

    def _open(self):
        if self.f224 is None:
            f=h5py.File(self.p224,"r"); self.f224=f
            self.m224=("images" in f and "keys" in f)
            if self.m224:
                try: keys=f["keys"].asstr()[()]
                except Exception: keys=[k.decode() for k in f["keys"][()]]
                self.k2i224={str(k):i for i,k in enumerate(keys)}
        if self.f1024 is None:
            f=h5py.File(self.p1024,"r"); self.f1024=f
            self.m1024=("images" in f and "keys" in f)
            if self.m1024:
                try: keys=f["keys"].asstr()[()]
                except Exception: keys=[k.decode() for k in f["keys"][()]]
                self.k2i1024={str(k):i for i,k in enumerate(keys)}

    def close(self):
        for f in [self.f224,self.f1024]:
            try:
                if f is not None: f.close()
            except Exception: pass
        self.f224=self.f1024=None

    @staticmethod
    def _fix(a: np.ndarray) -> np.ndarray:
        if a.ndim==2: a=np.stack([a]*3,-1)
        if a.ndim==3 and a.shape[-1]==4: a=a[...,:3]
        return np.clip(a,0,255).astype(np.uint8)

    def get_batch(self, ids: List[str]) -> Tuple[np.ndarray,np.ndarray]:
        self._open()
        # 224
        if self.m224:
            idx=[self.k2i224[str(k)] for k in ids]
            a224=self.f224["images"][idx]
        else:
            a224=np.stack([self.f224[str(k)][()] for k in ids],0)
        a224=np.stack([self._fix(x) for x in a224],0)
        if a224.shape[1:4]!=(224,224,3):
            raise ValueError(f"[224] shape wrong: {a224.shape}")
        # 1024
        if self.m1024:
            ds=self.f1024["images"]; H,W=int(ds.shape[1]), int(ds.shape[2])
            fH, fW = H//self.ctx, W//self.ctx
            can_stride = (fH*self.ctx==H and fW*self.ctx==W and getattr(ds,"compression",None) is None and getattr(ds,"chunks",None) is None)
            idx=[self.k2i1024[str(k)] for k in ids]
            if can_stride:
                a1024 = ds[idx, ::fH, ::fW, :]
            else:
                a1024 = ds[idx]
        else:
            a1024=np.stack([self.f1024[str(k)][()] for k in ids],0)
        a1024=np.stack([self._fix(x) for x in a1024],0)
        return a224, a1024

# ============== Label alignment helpers ==============
def _pick_label_column(df: pd.DataFrame) -> str:
    for k in ["class_id","label","cell_type","type","annotation"]:
        if k in df.columns: return k
    raise KeyError("annotations.json is missing class_id/label/cell_type/type/annotation.")

def build_xenium_exact_df(ann_json: str, train_class_names: List[str]) -> Tuple[pd.DataFrame, List[str], List[int]]:
    print(f"  - Reading annotations from {ann_json}")
    df=pd.read_json(ann_json, orient="index").reset_index().rename(columns={"index":"cell_id"})
    lab=_pick_label_column(df); df["orig"]=df[lab].astype(str)
    df=df[~df["orig"].isin(EXACT_REMOVE)].copy()
    df["norm"]=df["orig"].map(SYN_ALIAS).fillna(df["orig"])
    df["aligned"]=df["norm"].map(EXACT_HOLDOUT_TO_TRAIN).fillna(df["norm"])
    present=set(train_class_names); used=sorted([x for x in df["aligned"].unique() if x in present])
    c2i_used={n:i for i,n in enumerate(used)}
    idx_in_train=[train_class_names.index(n) for n in used]
    df=df[df["aligned"].isin(used)].copy()
    df["label"]=df["aligned"].map(c2i_used).astype(int)
    print(f"  - Samples={len(df)} | evaluation classes={len(used)}")
    return df[["cell_id","label","aligned"]], used, idx_in_train

def build_lizard_group_df(ann_json: str, train_class_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    print(f"  - Reading annotations from {ann_json}")
    df=pd.read_json(ann_json, orient="index").reset_index().rename(columns={"index":"cell_id"})
    lab=_pick_label_column(df); df["raw"]=df[lab].astype(str).str.strip().str.lower()
    valid=set(LIZARD_RAW_TO_GROUP.keys()); df=df[df["raw"].isin(valid)].copy()
    df["group"]=df["raw"].map(LIZARD_RAW_TO_GROUP)
    used=["Epithelial","Connective","Lymphocyte"]; c2i={n:i for i,n in enumerate(used)}
    df["label"]=df["group"].map(c2i).astype(int)
    # Group -> training-class probability projection matrix
    t2i={n:i for i,n in enumerate(train_class_names)}
    M=np.zeros((len(train_class_names), len(used)), dtype=np.float32)
    for gi,g in enumerate(used):
        for t in LIZARD_GROUP_TO_TRAIN[g]:
            if t in t2i: M[t2i[t], gi]=1.0
    row = M.sum(1, keepdims=True); row[row==0]=1.0
    M = M/row
    print(f"  - Samples={len(df)} | evaluation groups={used}")
    return df[["cell_id","label","group"]], M, used

# ============== Model (same submodules as training) ==============
class FiLM(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int):
        super().__init__()
        hid=max(64, in_dim*2)
        self.net=nn.Sequential(nn.Linear(in_dim,hid), nn.SiLU(), nn.Linear(hid, 2*feat_dim))
    def forward(self, t): return self.net(t).chunk(2, dim=-1)

class PathBInfer(nn.Module):
    def __init__(self, num_classes: int, num_tissues: int, dino_model_name: str, ctx_mb: int=32):
        super().__init__()
        self.ctx_mb=int(ctx_mb)

        # UNI encoder (same as training)
        uni_cfg={'img_size':224,'patch_size':14,'depth':24,'num_heads':24,'init_values':1e-5,'embed_dim':1536,
                 'mlp_ratio':2.66667*2,'num_classes':0,'no_embed_class':True,
                 'mlp_layer':timm.layers.SwiGLUPacked,'act_layer':torch.nn.SiLU,
                 'reg_tokens':8,'dynamic_img_size':True}
        self.uni_encoder=timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=False, **uni_cfg)
        self.uni_dim=self.uni_encoder.embed_dim

        # A-head stack (same names as training)
        self.a_feat_norm   = nn.LayerNorm(self.uni_dim)
        self.a_tissue_embed= nn.Embedding(num_tissues, 64)
        self.a_tissue_film = FiLM(64, self.uni_dim)
        self.a_head        = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.uni_dim,1024), nn.SiLU(), nn.Dropout(0.2), nn.Linear(1024, num_classes))

        # DINO branch (identical dino_model naming)
        resolved_dino = resolve_dino_model_path(dino_model_name, REPO_ROOT)
        self.dino_model = AutoModel.from_pretrained(
            resolved_dino,
            trust_remote_code=True,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            low_cpu_mem_usage=True,
        )
        self.dino_dim = int(self.dino_model.config.hidden_size)
        for p in self.dino_model.parameters(): p.requires_grad=False
        self.dino_model.eval()

        # Projection + B-head (matching training)
        self.proj_uni = nn.Sequential(nn.LayerNorm(self.uni_dim), nn.Linear(self.uni_dim, 512))
        self.proj_ctx = nn.Sequential(nn.LayerNorm(self.dino_dim), nn.Linear(self.dino_dim, 512))
        self.head_B   = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024,1024), nn.SiLU(), nn.Dropout(0.2), nn.Linear(1024, num_classes))

        # Gate MLP (kept for checkpoint compatibility)
        self.gate_mlp = nn.Sequential(nn.Linear(1025,256), nn.SiLU(), nn.Linear(256,1), nn.Sigmoid())

    # ---- Load Lightning checkpoint selectively (fix FiLM key name differences) ----
    @torch.no_grad()
    def load_from_lightning_ckpt(self, ckpt_path: str, map_location="cpu"):
        obj=torch.load(ckpt_path, map_location=map_location)
        state=obj.get("state_dict", obj)
        # Gather only the prefixes we care about
        wanted_prefixes = [
            "uni_encoder", "a_feat_norm", "a_tissue_embed", "a_tissue_film",
            "a_head", "dino_model", "proj_uni", "proj_ctx", "head_B", "gate_mlp"
        ]
        sub={}
        for k,v in state.items():
            if any(k.startswith(p+".") for p in wanted_prefixes):
                # Normalize legacy FiLM key variants such as a_tissue_film.0/2.*
                if k.startswith("a_tissue_film.0."):
                    k = k.replace("a_tissue_film.0.", "a_tissue_film.net.0.")
                if k.startswith("a_tissue_film.2."):
                    k = k.replace("a_tissue_film.2.", "a_tissue_film.net.2.")
                sub[k]=v

        # Load each sub-module and log stats
        def _load_sub(module: nn.Module, prefix: str):
            part={kk[len(prefix)+1:]:vv for kk,vv in sub.items() if kk.startswith(prefix+".")}
            missing, unexpected = module.load_state_dict(part, strict=False)
            print(f"[ckpt] {prefix:14s} | loaded={len(part):4d}  missing={len(missing):2d}  unexpected={len(unexpected):2d}")

        _load_sub(self.uni_encoder, "uni_encoder")
        _load_sub(self.a_feat_norm,  "a_feat_norm")
        _load_sub(self.a_tissue_embed, "a_tissue_embed")
        _load_sub(self.a_tissue_film,  "a_tissue_film")
        _load_sub(self.a_head, "a_head")
        _load_sub(self.dino_model, "dino_model")
        _load_sub(self.proj_uni, "proj_uni")
        _load_sub(self.proj_ctx, "proj_ctx")
        _load_sub(self.head_B, "head_B")
        _load_sub(self.gate_mlp, "gate_mlp")
        print("[ckpt] load done.")

    # ---- Forward pass (B logits + fused embedding; A-only kept for alignment) ----
    @torch.no_grad()
    def forward(self, x224: torch.Tensor, xctx: torch.Tensor, tissues: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_uni = self.uni_encoder(x224)                     # [N,1536]
        # A-head (optional for gate alignment; not returned)
        t = self.a_tissue_embed(tissues)
        gamma, beta = self.a_tissue_film(t)
        featA = self.a_feat_norm(feat_uni); featA = featA * (1+gamma) + beta
        _ = self.a_head(featA)

        # DINO micro-batch
        outs=[]; N=xctx.shape[0]
        for i in range(0, N, self.ctx_mb):
            xi=xctx[i:i+self.ctx_mb].to(dtype=self.dino_model.dtype)
            out=self.dino_model(pixel_values=xi, return_dict=True)
            vec= out.pooler_output if getattr(out,"pooler_output",None) is not None else out.last_hidden_state[:,1:,:].mean(1)
            outs.append(vec.float())
        ctx_vec=torch.cat(outs,0)                              # [N,dino_dim]

        uni_res=self.proj_uni(feat_uni)                        # [N,512]
        ctx_proj=self.proj_ctx(ctx_vec)                        # [N,512]
        fused=torch.cat([uni_res, ctx_proj], 1)                # [N,1024]
        zB=self.head_B(fused)                                  # [N,C]

        return zB, fused

# ============== Inference flow ==============
def infer_holdout(model: PathBInfer, ckpt_classes: List[str], name: str, root: str,
                  mode: str, tissue: str, out_dir: str, ctx_size: int=512, batch: int=256):
    os.makedirs(out_dir, exist_ok=True)
    ann = os.path.join(root,"annotations.json")
    p224= os.path.join(root,"patches_224x224.h5")
    p1024=os.path.join(root,"patches_1024x1024.h5")
    if not os.path.exists(ann):  raise FileNotFoundError(ann)
    if not os.path.exists(p224): raise FileNotFoundError(p224)
    if not os.path.exists(p1024):raise FileNotFoundError(p1024)

    # Build evaluation label space
    if mode=="xenium_exact":
        df_eval, used_names, idx_in_train = build_xenium_exact_df(ann, ckpt_classes)
        used_idx = torch.as_tensor(idx_in_train, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
        M = None
    elif mode=="lizard_grouped":
        df_eval, M, used_names = build_lizard_group_df(ann, ckpt_classes)
        used_idx=None
        M = torch.from_numpy(M).to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    else:
        raise ValueError(mode)

    with open(os.path.join(out_dir, "class_names_used.json"), "w") as f:
        json.dump(used_names, f, indent=2)

    rdr=H5DualReader(p224, p1024, ctx_size)
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    organ_id = TISSUE_TO_IDX.get(tissue, 0)

    all_emb=[]; all_probs=[]; all_labels=[]; all_ids=[]
    progress = tqdm(
        range(0, len(df_eval), batch),
        desc=f"[{name}] ctx={ctx_size} batch={batch}",
        unit="batch",
        leave=False,
    )
    for s in progress:
        part=df_eval.iloc[s:s+batch]
        ids=part["cell_id"].astype(str).tolist()
        a224,a1024=rdr.get_batch(ids)
        x224 = torch.from_numpy(a224).permute(0,3,1,2)
        x1024= torch.from_numpy(a1024).permute(0,3,1,2)

        x224  = norm_img_uint8_to_float(x224, dev)
        if x1024.shape[-2:]!=(ctx_size,ctx_size):
            xctx=F.interpolate(x1024.to(device=dev, dtype=torch.float32).div_(255.0),
                               size=(ctx_size,ctx_size), mode="bilinear", align_corners=False)
            xctx=xctx.sub_(IM_MEAN.to(dev)).div_(IM_STD.to(dev))
        else:
            xctx = norm_img_uint8_to_float(x1024, dev)

        tissues=torch.full((x224.shape[0],), organ_id, dtype=torch.long, device=dev)

        zB, emb = model(x224, xctx, tissues)          # [N,C], [N,1024]
        p_train = torch.softmax(zB,1).to(dtype=torch.float32)

        if mode=="xenium_exact":
            probs = p_train.index_select(1, used_idx)
        else:
            probs = p_train @ M

        all_emb.append(emb.cpu()); all_probs.append(probs.cpu())
        all_labels.append(torch.tensor(part["label"].values, dtype=torch.long))
        all_ids.extend(ids)

    rdr.close()

    emb = torch.cat(all_emb).numpy()
    p   = torch.cat(all_probs).numpy()
    y   = torch.cat(all_labels).numpy()
    yhat= p.argmax(1)

    # Persist outputs (aligned with A/PLIP/LOKI expectations)
    np.save(os.path.join(out_dir,"embeddings.npy"), emb)
    np.save(os.path.join(out_dir,"probe.npy"),       emb)
    np.save(os.path.join(out_dir,"probs.npy"),       p)
    np.save(os.path.join(out_dir,"labels.npy"),      y)
    np.save(os.path.join(out_dir,"preds.npy"),       yhat)

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc=float(accuracy_score(y,yhat)); f1m=float(f1_score(y,yhat,average="macro",zero_division=0))
    with open(os.path.join(out_dir,"metrics.json"),"w") as f:
        json.dump({"overall_acc":acc,"macro_f1":f1m,"n_samples":int(len(y)),"classes_eval":used_names}, f, indent=2)
    print(f"[{name}] acc={acc:.4f}  f1_macro={f1m:.4f}")

    # Optional confusion matrix plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, seaborn as sns
        K=len(used_names); cm=confusion_matrix(y,yhat,labels=list(range(K)),normalize="true")
        plt.figure(figsize=(max(10,K*0.7),8))
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", xticklabels=used_names, yticklabels=used_names)
        plt.title(f"Confusion Matrix ({name})"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"confusion_matrix.png"), dpi=200)
        plt.close()
    except Exception:
        pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="Path-B Lightning ckpt")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directory to store inference outputs.")
    ap.add_argument("--holdouts-config", type=str, default=str(DEFAULT_HOLDOUTS_FILE),
                    help="JSON file describing holdouts (root/mode/tissue).")
    ap.add_argument("--select", nargs="+", default=None, help="Optional subset of holdout names to run.")
    ap.add_argument("--ctx_size", type=int, default=512)
    ap.add_argument("--mb", type=int, default=32, help="DINO micro-batch")
    ap.add_argument("--batch", type=int, default=256)
    args=ap.parse_args()

    holdouts = load_holdouts_config(args.holdouts_config)
    selected = args.select or list(holdouts.keys())
    ckpt_path = str(Path(args.ckpt).expanduser().resolve())

    # Always use cuda:0 (respecting any CUDA_VISIBLE_DEVICES mask)
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev.type=="cuda":
        print("[CUDA]", torch.cuda.get_device_name(dev))

    raw=torch.load(ckpt_path, map_location="cpu")
    hp = raw.get("hyper_parameters", {})
    class_names = hp.get("class_names", None)
    num_tissues = int(hp.get("num_tissues", 8))
    dino_name   = hp.get("config",{}).get("dino_model","facebook/dinov3-vitl16-pretrain-lvd1689m")
    if not class_names:
        raise RuntimeError("Checkpoint missing hyper_parameters.class_names; export ckpt with training metadata.")

    model=PathBInfer(num_classes=len(class_names), num_tissues=num_tissues, dino_model_name=dino_name, ctx_mb=args.mb)
    model.load_from_lightning_ckpt(ckpt_path, map_location="cpu")
    model.to(dev).eval()

    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for name in selected:
        if name not in holdouts:
            raise KeyError(f"Holdout '{name}' not found in {args.holdouts_config}")
        info = holdouts[name]
        out_dir=os.path.join(out_root, name); os.makedirs(out_dir, exist_ok=True)
        print(f"\n[holdout] {name} | mode={info['mode']}")
        infer_holdout(model, class_names, name, info["root"], info["mode"], info["tissue"],
                      out_dir, ctx_size=args.ctx_size, batch=args.batch)

if __name__=="__main__":
    main()
