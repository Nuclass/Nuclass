# -*- coding: utf-8 -*-
"""
FuseAB GateMix inference (aligned with the Path-A evaluation recipe)
- Restores modelA/modelB/gate from the fusion checkpoint with FiLM key compatibility.
- Prefers class_names from ckptA (hyper_parameters.ckptA) and falls back to ckptB.
- Evaluation mode: Xenium (ovary/lung/pancreas) with strict class mapping (synonym normalization, drop SMC/Pericyte + Myeloid).
- Outputs per-class F1/ACC for the mixed logits, confusion matrix PNG, and metrics.json.
"""

import os, re, json, argparse, warnings, h5py
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Environment + quiet mode
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
warnings.filterwarnings("ignore", message=r".*No device id is provided.*")

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
from transformers import AutoModel, logging as hf_logging
hf_logging.set_verbosity_error()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "path_gate_release" / "best.ckpt"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "test" / "gate_inference"
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

# ---------------- Label normalization (matches Path-A inference) ----------------
def _canon_label(x: str) -> str:
    s = str(x)
    s = re.sub(r"\s+", " ", s.strip())
    s = s.replace("\u2013","-").replace("\u2014","-")
    alt = s.replace("_"," ") if "_" in s and " " not in s else (s.replace(" ","_") if " " in s and "_" not in s else s)
    SYN = {
        "Fibroblasts":"Fibroblast", "fibroblasts":"Fibroblast",
        "T cells":"T_Cell", "T cell":"T_Cell", "T Cells":"T_Cell", "T Cell":"T_Cell", "T_cells":"T_Cell",
        "Endothelial Cell":"Endothelial", "Endothelial cell":"Endothelial",
        "Macrophages":"Macrophage", "macrophages":"Macrophage",
        "Tumor epithelial":"Epithelial_Malignant", "Tumor_epithelial":"Epithelial_Malignant",
        "SMC/Pericytes":"SMC/Pericyte",
        "Stromal Associated Fibroblasts":"Tumor_Associated_Fibroblast",
        "Tumor Associated Fibroblast":"Tumor_Associated_Fibroblast",
        "Tumor Associated Fibroblasts":"Tumor_Associated_Fibroblast",
    }
    return SYN.get(s, SYN.get(alt, s))

EXACT_REMOVE = {"SMC/Pericyte", "Myeloid"}
EXACT_TO_TRAIN = {
    "Epithelial_Malignant":"Epithelial_Malignant",
    "Fibroblast":"Fibroblast", "Ductal":"Ductal", "Endothelial":"Endothelial", "Macrophage":"Macrophage",
    "Acinar":"Acinar", "Endocrine":"Endocrine", "Tumor_Associated_Fibroblast":"Tumor_Associated_Fibroblast",
    "Monocyte":"Monocyte", "T_Cell":"T_Cell", "B_Cell":"B_Cell", "NK_Cell":"NK_Cell", "Plasma_Cell":"Plasma_Cell",
}

# ---------------- Image normalization ----------------
IM_MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
IM_STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
def norm_img_uint8_to_float(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device=device, dtype=torch.float32).div_(255.0)
    return x.sub_(IM_MEAN.to(device)).div_(IM_STD.to(device))

# ---------------- H5 224/1024 readers (with stride downsampling) ----------------
class H5DualReader:
    def __init__(self, p224: str, p1024: str, ctx: int):
        self.p224, self.p1024, self.ctx = p224, p1024, int(ctx)
        for p in [p224,p1024]:
            if not os.path.exists(p): raise FileNotFoundError(p)
        self.f224=self.f1024=None; self.m224=self.m1024=False; self.k2i224=self.k2i1024=None

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
        if a224.shape[1:4]!=(224,224,3): raise ValueError(f"[224] shape wrong: {a224.shape}")

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

    def close(self):
        for f in [self.f224,self.f1024]:
            try:
                if f is not None: f.close()
            except Exception: pass
        self.f224=self.f1024=None

# ---------------- Evaluation label-space builders ----------------
def _pick_label_column(df: pd.DataFrame) -> str:
    for k in ["class_id","label","cell_type","type","annotation"]:
        if k in df.columns: return k
    raise KeyError("annotations.json is missing class_id/label/cell_type/type/annotation.")

def build_xenium_exact_df(ann_json: str, train_names: List[str]) -> Tuple[pd.DataFrame, List[str], List[int]]:
    print(f"  - Reading annotations from {ann_json}")
    df=pd.read_json(ann_json, orient="index").reset_index().rename(columns={"index":"cell_id"})
    lab=_pick_label_column(df)
    df["orig"]=df[lab].astype(str).apply(_canon_label)
    df=df[~df["orig"].isin(EXACT_REMOVE)].copy()
    df["aligned"]=df["orig"].map(EXACT_TO_TRAIN).fillna(df["orig"])

    present=set(train_names)
    used=sorted([x for x in df["aligned"].unique() if x in present])
    miss=[x for x in df["aligned"].unique() if x not in present]
    if miss:
        print("  - Dropped (not present in training label space):")
        for k,v in df.loc[df["aligned"].isin(miss),"aligned"].value_counts().items():
            print(f"      {k:28s} : {v}")
    df=df[df["aligned"].isin(used)].copy()

    c2i={n:i for i,n in enumerate(used)}
    df["label"]=df["aligned"].map(c2i).astype(int)

    used_index_in_train=[train_names.index(n) for n in used]
    print("  - Column alignment (used -> train_col):")
    for n,i in zip(used, used_index_in_train):
        print(f"      {n:28s} -> train_col {i}")
    print(f"  - Samples={len(df)} | evaluation classes={len(used)}")
    return df[["cell_id","label","aligned"]], used, used_index_in_train

# ---------------- Model definitions (training-aligned + FiLM-compatible) ----------------
# timm MLP compatibility
try:
    from timm.layers import SwiGLUPacked as _MLP
except Exception:
    from timm.layers import SwiGLU as _MLP

class FiLM(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super().__init__()
        hid=max(64, in_dim*2)
        self.net=nn.Sequential(nn.Linear(in_dim,hid), nn.SiLU(), nn.Linear(hid, 2*feat_dim))
    def forward(self, t): return self.net(t).chunk(2, dim=-1)

class PathA_InferExact(nn.Module):
    def __init__(self, num_classes, num_tissues, tdim=64):
        super().__init__()
        uni_cfg={'img_size':224,'patch_size':14,'depth':24,'num_heads':24,'init_values':1e-5,
                 'embed_dim':1536,'mlp_ratio':2.66667*2,'num_classes':0,'no_embed_class':True,
                 'mlp_layer':_MLP,'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True}
        self.uni_encoder=timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)
        self.feat_norm=nn.LayerNorm(self.uni_encoder.embed_dim)
        self.tissue_embedder=nn.Embedding(num_tissues, tdim)
        self.tissue_film=FiLM(tdim, self.uni_encoder.embed_dim)
        self.head=nn.Sequential(nn.Dropout(0.2), nn.Linear(self.uni_encoder.embed_dim,1024),
                                nn.SiLU(), nn.Dropout(0.2), nn.Linear(1024, num_classes))
    def forward(self, x224, tissue, return_feat=False):
        feat=self.uni_encoder(x224)
        feat=self.feat_norm(feat)
        t=self.tissue_embedder(tissue)
        g,b=self.tissue_film(t)
        feat=feat*(1+g)+b
        logits=self.head(feat)
        return (logits, feat) if return_feat else logits

class PathB_InferExact(nn.Module):
    def __init__(self, num_classes, dino_model, use_pooler=True, ctx_microbatch=16):
        super().__init__()
        self.use_pooler=use_pooler; self.ctx_microbatch=int(ctx_microbatch)
        uni_cfg={'img_size':224,'patch_size':14,'depth':24,'num_heads':24,'init_values':1e-5,
                 'embed_dim':1536,'mlp_ratio':2.66667*2,'num_classes':0,'no_embed_class':True,
                 'mlp_layer':_MLP,'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True}
        self.uni_encoder=timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)
        self.proj_uni=nn.Sequential(nn.LayerNorm(self.uni_encoder.embed_dim), nn.Linear(self.uni_encoder.embed_dim, 512))

        resolved_dino = resolve_dino_model_path(dino_model, REPO_ROOT)
        self.dino_model=AutoModel.from_pretrained(
            resolved_dino,
            trust_remote_code=True,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            low_cpu_mem_usage=True
        )
        self.dino_dim=int(self.dino_model.config.hidden_size)
        self.proj_ctx=nn.Sequential(nn.LayerNorm(self.dino_dim), nn.Linear(self.dino_dim, 512))
        self.head=nn.Sequential(nn.Dropout(0.2), nn.Linear(1024,1024), nn.SiLU(), nn.Dropout(0.2), nn.Linear(1024, num_classes))

    @torch.no_grad()
    def _ctx_vec(self, xctx):
        outs=[]; N=xctx.shape[0]; mb=max(1, min(self.ctx_microbatch, N))
        for i in range(0,N,mb):
            xi=xctx[i:i+mb].to(dtype=self.dino_model.dtype)
            out=self.dino_model(pixel_values=xi, return_dict=True)
            vec= out.pooler_output if getattr(out,"pooler_output",None) is not None else out.last_hidden_state[:,1:,:].mean(1)
            outs.append(vec.float())
        return torch.cat(outs,0)
    def forward(self, x224, xctx):
        u=self.uni_encoder(x224)
        v=self._ctx_vec(xctx)
        u_proj=self.proj_uni(u); v_proj=self.proj_ctx(v)
        fused=torch.cat([u_proj, v_proj], 1)
        z=self.head(fused)
        return z, fused

class FuseABGateInfer(nn.Module):
    def __init__(self, num_classes, num_tissues, dino_model, gate_dim_A=128, gate_dim_B=128, ctx_mb=16):
        super().__init__()
        self.modelA = PathA_InferExact(num_classes, num_tissues, tdim=64)
        self.modelB = PathB_InferExact(num_classes, dino_model=dino_model, use_pooler=True, ctx_microbatch=ctx_mb)

        # Gate input: proj_A(featA) + proj_B(fusedB) + 8 statistics
        self.gate_proj_A = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, gate_dim_A), nn.SiLU(), nn.Dropout(0.1))
        self.gate_proj_B = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, gate_dim_B), nn.SiLU(), nn.Dropout(0.1))
        in_dim = gate_dim_A + gate_dim_B + 8
        self.gate_mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.SiLU(), nn.Dropout(0.1),
                                      nn.Linear(256, 64), nn.SiLU(), nn.Dropout(0.1),
                                      nn.Linear(64, 1))

    @torch.no_grad()
    def load_from_fuse_ckpt(self, fuse_ckpt: str, map_location="cpu"):
        obj=torch.load(fuse_ckpt, map_location=map_location)
        sd=obj.get("state_dict", obj)

        # FiLM key compatibility (legacy checkpoints)
        def _fix_key(k: str) -> str:
            k = k.replace("modelA.tissue_film.0.", "modelA.tissue_film.net.0.")
            k = k.replace("modelA.tissue_film.2.", "modelA.tissue_film.net.2.")
            k = k.replace("modelB.proj_ctx.0.", "modelB.proj_ctx.0.")  # kept for clarity
            return k

        sd = { _fix_key(k):v for k,v in sd.items() }

        def _load_prefix(module: nn.Module, prefix: str):
            sub={kk[len(prefix)+1:]:vv for kk,vv in sd.items() if kk.startswith(prefix+".")}
            missing, unexpected = module.load_state_dict(sub, strict=False)
            print(f"[ckpt] {prefix:15s} | loaded={len(sub):4d}  missing={len(missing):2d}  unexpected={len(unexpected):2d}")

        _load_prefix(self.modelA, "modelA")
        _load_prefix(self.modelB, "modelB")
        _load_prefix(self.gate_proj_A, "gate_proj_A")
        _load_prefix(self.gate_proj_B, "gate_proj_B")
        _load_prefix(self.gate_mlp,    "gate_mlp")
        print("[ckpt] load done.")

    @staticmethod
    def _stats_from_probs(p: torch.Tensor):
        # Normalized entropy (log scaled by class count)
        C=p.shape[1]
        pmax=p.max(1).values
        top2,_=p.topk(k=min(2,C),dim=1)
        margin=top2[:,0] - (top2[:,1] if C>1 else 0.0)
        ent=-(p.clamp_min(1e-8).log()*p).sum(1) / np.log(max(C,2))
        return pmax, ent, margin

    @torch.no_grad()
    def forward(self, x224: torch.Tensor, xctx: torch.Tensor, tissue_ids: torch.Tensor):
        # Normalization handled externally
        zA, featA = self.modelA(x224, tissue_ids, return_feat=True)
        zB, fusedB = self.modelB(x224, xctx)

        pA=torch.softmax(zA.float(),1)
        pB=torch.softmax(zB.float(),1)

        pAmax, entA, marA = self._stats_from_probs(pA)
        pBmax, entB, marB = self._stats_from_probs(pB)
        delta = (pBmax - pAmax)

        fA = self.gate_proj_A(featA.float())
        fB = self.gate_proj_B(fusedB.float())
        gate_in = torch.cat([
            fA, fB,
            pAmax[:, None], pBmax[:, None],
            entA[:, None],  entB[:, None],
            marA[:, None],  marB[:, None],
            delta[:, None], delta.abs()[:, None],
        ], dim=1)        
        g_logit=self.gate_mlp(gate_in).squeeze(1)
        g=torch.sigmoid(g_logit)

        p_mix = (1.0 - g)[:,None]*pA + g[:,None]*pB
        return pA, pB, p_mix, g, fusedB

# ---------------- Safety gating (when safety_policy.json exists) ----------------
def _reliability_by_pred_class(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred=probs.argmax(1); C=probs.shape[1]; rel=np.zeros(C, dtype=np.float32)
    for c in range(C):
        idx=np.where(pred==c)[0]
        if idx.size>0: rel[c]=(y[idx]==c).mean()
    return rel

def _baseline_choose(a_probs: np.ndarray, b_probs: np.ndarray, relA: np.ndarray, relB: np.ndarray) -> np.ndarray:
    a_max=a_probs.max(1); a_cls=a_probs.argmax(1)
    b_max=b_probs.max(1); b_cls=b_probs.argmax(1)
    scoreA=a_max*(relA[a_cls]+1e-6)
    scoreB=b_max*(relB[b_cls]+1e-6)
    chooseB=scoreB>scoreA
    return np.where(chooseB[:,None], b_probs, a_probs)

def _apply_safe_policy(a_probs, b_probs, g, relA, relB, tau, gamma):
    a_max=a_probs.max(1); b_max=b_probs.max(1)
    base=_baseline_choose(a_probs,b_probs,relA,relB)
    chooseB = (g>tau) & ((b_max - a_max) > gamma)
    chooseA = (g<(1-tau)) & ((a_max - b_max) > gamma)
    out=base.copy(); out[chooseB]=b_probs[chooseB]; out[chooseA]=a_probs[chooseA]
    return out

# ---------------- per-class metrics & confusion matrix ----------------
def save_per_class_and_cm(y_true: np.ndarray, y_pred: np.ndarray, names: List[str], out_dir: str, prefix="aligned_mix"):
    from sklearn.metrics import f1_score, confusion_matrix
    K=len(names)
    per_f1 = f1_score(y_true, y_pred, labels=list(range(K)), average=None, zero_division=0)
    per_acc=[]
    for c in range(K):
        m=(y_true==c); per_acc.append(float((y_pred[m]==c).mean()) if m.any() else 0.0)
    pd.DataFrame({"class":names,"f1":per_f1}).to_csv(os.path.join(out_dir, f"{prefix}_per_class_f1.csv"), index=False)
    pd.DataFrame({"class":names,"acc":per_acc}).to_csv(os.path.join(out_dir, f"{prefix}_per_class_acc.csv"), index=False)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, seaborn as sns
        cm=confusion_matrix(y_true,y_pred,labels=list(range(K)),normalize='true')
        plt.figure(figsize=(max(10,K*0.7),8))
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", xticklabels=names, yticklabels=names)
        plt.title("Confusion Matrix (GateMix aligned)"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"), dpi=200); plt.close()
    except Exception:
        pass

# ---------------- Inference for a single holdout ----------------
@torch.no_grad()
def infer_holdout(model: FuseABGateInfer, train_classes: List[str], name: str, root: str,
                  mode: str, tissue: str, out_dir: str, ctx_size: int=384, batch: int=128):
    os.makedirs(out_dir, exist_ok=True)
    ann=os.path.join(root,"annotations.json")
    p224=os.path.join(root,"patches_224x224.h5")
    p1024=os.path.join(root,"patches_1024x1024.h5")
    if not os.path.exists(ann):  raise FileNotFoundError(ann)
    if not os.path.exists(p224): raise FileNotFoundError(p224)
    if not os.path.exists(p1024):raise FileNotFoundError(p1024)

    if mode != "xenium_exact":
        raise ValueError(f"Unsupported mode '{mode}'. Only 'xenium_exact' is available.")
    df_eval, used_names, idx_in_train = build_xenium_exact_df(ann, train_classes)
    used_idx = torch.as_tensor(idx_in_train, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(out_dir,"class_names_used.json"),"w") as f: json.dump(used_names, f, indent=2)

    rdr=H5DualReader(p224,p1024,ctx_size)
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    organ_id=TISSUE_TO_IDX.get(tissue,0)

    all_emb=[]; all_mix=[]; all_A=[]; all_B=[]; all_g=[]; all_y=[]

    print(f"[{name}] ctx={ctx_size} batch={batch}")
    for s in tqdm(range(0,len(df_eval),batch), desc=f"[{name}] infer", ncols=100):
        part=df_eval.iloc[s:s+batch]
        ids=part["cell_id"].astype(str).tolist()
        a224,a1024=rdr.get_batch(ids)

        x224=torch.from_numpy(a224).permute(0,3,1,2)
        x1024=torch.from_numpy(a1024).permute(0,3,1,2)

        x224 = norm_img_uint8_to_float(x224, dev)
        if x1024.shape[-2:]!=(ctx_size,ctx_size):
            xctx=F.interpolate(x1024.to(device=dev, dtype=torch.float32).div_(255.0),
                               size=(ctx_size,ctx_size), mode="bilinear", align_corners=False)
            xctx=xctx.sub_(IM_MEAN.to(dev)).div_(IM_STD.to(dev))
        else:
            xctx = norm_img_uint8_to_float(x1024, dev)

        tissue_ids=torch.full((x224.shape[0],), organ_id, dtype=torch.long, device=dev)

        with torch.autocast(device_type=("cuda" if dev.type=="cuda" else "cpu"), dtype=torch.bfloat16, enabled=(dev.type=="cuda")):
            pA,pB,pM,g,emb = model(x224,xctx,tissue_ids)
            pA=pA.to(dtype=torch.float32); pB=pB.to(dtype=torch.float32); pM=pM.to(dtype=torch.float32)

            probs_A = pA.index_select(1, used_idx)
            probs_B = pB.index_select(1, used_idx)
            probs_M = pM.index_select(1, used_idx)

        all_A.append(probs_A.cpu()); all_B.append(probs_B.cpu()); all_mix.append(probs_M.cpu())
        all_g.append(g.detach().float().cpu()); all_emb.append(emb.float().cpu())
        all_y.append(torch.tensor(part["label"].values, dtype=torch.long))

    rdr.close()

    probs_A = torch.cat(all_A).numpy()
    probs_B = torch.cat(all_B).numpy()
    probs_M = torch.cat(all_mix).numpy()
    gates   = torch.cat(all_g).numpy()
    emb     = torch.cat(all_emb).numpy()
    y       = torch.cat(all_y).numpy()
    yhat    = probs_M.argmax(1)

    # Persist outputs (mix focus; also save A/B branches and gates/embeddings)
    np.save(os.path.join(out_dir,"probs_A.npy"), probs_A)
    np.save(os.path.join(out_dir,"probs_B.npy"), probs_B)
    np.save(os.path.join(out_dir,"probs_mix.npy"), probs_M)
    np.save(os.path.join(out_dir,"gates.npy"),   gates)
    np.save(os.path.join(out_dir,"embeddings.npy"), emb)
    np.save(os.path.join(out_dir,"probe.npy"),       emb)  # alias
    np.save(os.path.join(out_dir,"labels.npy"), y)
    np.save(os.path.join(out_dir,"preds.npy"),  yhat)

    from sklearn.metrics import accuracy_score, f1_score
    accA=float(accuracy_score(y, probs_A.argmax(1)))
    accB=float(accuracy_score(y, probs_B.argmax(1)))
    accM=float(accuracy_score(y, yhat))
    f1A=float(f1_score(y, probs_A.argmax(1), average="macro", zero_division=0))
    f1B=float(f1_score(y, probs_B.argmax(1), average="macro", zero_division=0))
    f1M=float(f1_score(y, yhat,            average="macro", zero_division=0))

    # Try loading safety_policy.json (if present) for global reference
    safe_acc=None; safe_f1=None
    ckpt_dir = os.path.dirname(os.path.dirname(out_dir))  # fallback; main() prefers explicit override
    safe_json = None
    try:
        # main() usually passes ckpt_dir; otherwise walk upward from out_dir
        for base in [ckpt_dir, os.path.dirname(ckpt_dir), os.path.dirname(os.path.dirname(ckpt_dir)), out_dir]:
            cand=os.path.join(base,"safety_policy.json")
            if os.path.exists(cand): safe_json=cand; break
        if safe_json:
            safe = json.load(open(safe_json,"r"))
            relA=np.array(safe["relA"]); relB=np.array(safe["relB"])
            probs_safe = _apply_safe_policy(probs_A, probs_B, gates.reshape(-1), relA, relB,
                                            float(safe.get("tau",0.5)), float(safe.get("gamma",0.05))) \
                         if safe.get("mode","baseline")=="safe" else _baseline_choose(probs_A, probs_B, relA, relB)
            safe_acc=float(accuracy_score(y, probs_safe.argmax(1)))
            safe_f1=float(f1_score(y, probs_safe.argmax(1), average="macro", zero_division=0))
            np.save(os.path.join(out_dir,"probs_safe.npy"), probs_safe)
    except Exception:
        pass

    with open(os.path.join(out_dir,"metrics.json"),"w") as f:
        json.dump({
            "acc_A":accA,"f1_A":f1A,
            "acc_B":accB,"f1_B":f1B,
            "acc_mix":accM,"f1_mix":f1M,
            "acc_safe":safe_acc,"f1_safe":safe_f1,
            "n_samples":int(len(y)),"classes_eval":list(map(str, used_names))
        }, f, indent=2)

    # per-class (mix) + confusion matrix
    save_per_class_and_cm(y, yhat, used_names, out_dir, prefix="aligned_mix")
    print(f"[{name}] A(acc={accA:.4f},f1={f1A:.4f}) | B(acc={accB:.4f},f1={f1B:.4f}) | Mix(acc={accM:.4f},f1={f1M:.4f})"
          + (f" | Safe(acc={safe_acc:.4f},f1={safe_f1:.4f})" if safe_acc is not None else ""))

# ---------------- Main entry ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="Fusion Gate ckpt (.ckpt from Lightning)")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                    help="Directory to store inference outputs.")
    ap.add_argument("--holdouts-config", type=str, default=str(DEFAULT_HOLDOUTS_FILE),
                    help="JSON file describing holdouts (root/mode/tissue).")
    ap.add_argument("--select", nargs="+", default=None, help="Optional subset of holdout names")
    ap.add_argument("--ctx_size", type=int, default=384)
    ap.add_argument("--mb", type=int, default=16, help="DINO micro-batch")
    ap.add_argument("--batch", type=int, default=128)
    args=ap.parse_args()

    holdouts = load_holdouts_config(args.holdouts_config)
    selected = args.select or list(holdouts.keys())
    ckpt_path = str(Path(args.ckpt).expanduser().resolve())

    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev.type=="cuda":
        print("[CUDA]", torch.cuda.get_device_name(dev))

    # Parse gate hyper-parameters to locate ckptA/ckptB and recover metadata
    raw=torch.load(ckpt_path, map_location="cpu")
    hp=raw.get("hyper_parameters", {})
    cfg=hp.get("config", {})
    ckptA = hp.get("ckptA") or cfg.get("ckpt_pathA")
    ckptB = hp.get("ckptB") or cfg.get("ckpt_pathB")
    if not ckptA and not ckptB:
        raise RuntimeError("Unable to find ckptA/ckptB paths inside the gate checkpoint hyper-parameters.")

    # Training class names prefer ckptA, then fall back to ckptB
    class_names=None
    for c in [ckptA, ckptB]:
        if not c: continue
        try:
            r=torch.load(c, map_location="cpu")
            class_names=r.get("hyper_parameters",{}).get("class_names", None)
            if class_names: break
        except Exception:
            continue
    if not class_names:
        raise RuntimeError("Failed to recover class_names from ckptA/ckptB.")

    num_tissues = int(hp.get("num_tissues", 8))
    dino_name   = cfg.get("dino_model","facebook/dinov3-vitl16-pretrain-lvd1689m")
    gdimA = int(cfg.get("gate_proj_dim_A", 128))
    gdimB = int(cfg.get("gate_proj_dim_B", 128))

    # Ensure tissue ordering matches training
    expected = ["breast","colon","heart","kidney","liver","lung","ovary","pancreas"]
    assert set(TISSUE_TO_IDX.keys())==set(expected) and all(TISSUE_TO_IDX[k]==i for i,k in enumerate(expected)), \
        f"TISSUE_TO_IDX must match the sorted training order: {expected}"

    # Build and load weights
    model=FuseABGateInfer(num_classes=len(class_names), num_tissues=num_tissues,
                          dino_model=dino_name, gate_dim_A=gdimA, gate_dim_B=gdimB, ctx_mb=args.mb)
    model.load_from_fuse_ckpt(ckpt_path, map_location="cpu")
    model.to(dev).eval()

    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for name in selected:
        if name not in holdouts:
            raise KeyError(f"Holdout '{name}' not found in {args.holdouts_config}")
        info=holdouts[name]
        out_dir=os.path.join(out_root, name); os.makedirs(out_dir, exist_ok=True)
        print(f"\n[holdout] {name} | mode={info['mode']}")
        infer_holdout(model, class_names, name, info["root"], info["mode"], info["tissue"],
                      out_dir, ctx_size=args.ctx_size, batch=args.batch)

if __name__=="__main__":
    main()
