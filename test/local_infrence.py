# -*- coding: utf-8 -*-
"""
Path-A inference (aligned & robust)
- Xenium: strict 1:1 class-space alignment after robust normalization
- Hard checks: class-name alignment assertions + explicit mapping print
- Outputs: embeddings/probe/logits/probs_used/y_true/y_pred/metrics + CM
"""

import os, json, warnings, multiprocessing as mp, re, argparse
from pathlib import Path
from typing import Dict, List, Tuple
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
warnings.filterwarnings("ignore", message=r".*No device id is provided.*")

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "path_local_release" / "best.ckpt"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "test" / "path_a_inference"
DEFAULT_HOLDOUTS_FILE = Path(__file__).resolve().parent / "configs" / "holdouts_template.json"

# **Must match the training-time ordering exactly**
TISSUE_TO_IDX = {"breast":0,"colon":1,"heart":2,"kidney":3,"liver":4,"lung":5,"ovary":6,"pancreas":7}


def load_holdouts_config(path: Path) -> Dict[str, Dict[str, str]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if isinstance(cfg, list):
        cfg = {item.pop("name"): item for item in cfg}
    if not isinstance(cfg, dict):
        raise ValueError("Holdouts config must be a dict or list of objects with 'name'.")
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

# -------- robust normalization helpers --------
def _canon_label(x: str) -> str:
    s = str(x)
    s = re.sub(r"\s+", " ", s.strip())                  # collapse repeating whitespace
    s = s.replace("\u2013","-").replace("\u2014","-")
    # Handle underscore/space swapping both directions
    if "_" in s and " " not in s:
        alt = s.replace("_"," ")
    elif " " in s and "_" not in s:
        alt = s.replace(" ", "_")
    else:
        alt = s

    # Normalize case while preserving underscore style
    s2 = s
    # Synonym map (loose matching for both `s` and `alt`)
    SYN = {
        "Fibroblasts": "Fibroblast", "fibroblasts":"Fibroblast",
        "T cells":"T_Cell", "T cell":"T_Cell", "T Cells":"T_Cell", "T Cell":"T_Cell", "T_cells":"T_Cell",
        "Endothelial Cell":"Endothelial", "Endothelial cell":"Endothelial",
        "Macrophages":"Macrophage", "macrophages":"Macrophage",
        "Tumor Associated Fibroblast":"Tumor_Associated_Fibroblast",
        "Tumor Associated Fibroblasts":"Tumor_Associated_Fibroblast",
        "Stromal Associated Fibroblasts":"Tumor_Associated_Fibroblast",
        "SMC/Pericytes":"SMC/Pericyte",
        "Tumor epithelial":"Epithelial_Malignant", "Tumor_epithelial":"Epithelial_Malignant",
    }
    return SYN.get(s, SYN.get(alt, s2))

# -------- exact 1-1 for Xenium --------
EXACT_REMOVE = {"SMC/Pericyte", "Myeloid"}  # drop to stay aligned with training
EXACT_TO_TRAIN_ID = {
    "Epithelial_Malignant":"Epithelial_Malignant",
    "Fibroblast":"Fibroblast", "Ductal":"Ductal", 
    "Endothelial":"Endothelial", 
    "Macrophage":"Macrophage",
    "Acinar":"Acinar", 
    "Endocrine":"Endocrine", 
    "Tumor_Associated_Fibroblast":"Tumor_Associated_Fibroblast",
    "Monocyte":"Monocyte", 
    "T_Cell":"T_Cell", 
    "B_Cell":"B_Cell", 
    "NK_Cell":"NK_Cell", 
    "Plasma_Cell":"Plasma_Cell",
}

# ---------------- model (Path-A) ----------------
from Nuclass.utils.torchvision_compat import ensure_torchvision_nms

ensure_torchvision_nms()
import timm
from pytorch_lightning import LightningModule

class FiLM(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int):
        super().__init__()
        hidden = max(64, in_dim * 2)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2*feat_dim))
    def forward(self, t):
        gamma, beta = self.net(t).chunk(2, dim=-1)
        return gamma, beta

class AOnlyModule(LightningModule):
    def __init__(self, num_classes, class_names, num_tissues, tissue_dim: int = 64):
        super().__init__()
        self.class_names = list(class_names); self.num_classes = int(num_classes)
        uni_cfg = {'img_size':224, 'patch_size':14, 'depth':24, 'num_heads':24, 'init_values':1e-5,
                   'embed_dim':1536, 'mlp_ratio':2.66667*2, 'num_classes':0, 'no_embed_class':True,
                   'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 'reg_tokens':8, 'dynamic_img_size':True}
        self.uni_encoder = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)
        self.feat_norm   = nn.LayerNorm(self.uni_encoder.embed_dim)
        self.tissue_embedder = nn.Embedding(num_tissues, tissue_dim)
        self.tissue_film     = FiLM(tissue_dim, self.uni_encoder.embed_dim)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.uni_encoder.embed_dim, 1024), nn.SiLU(),
                                  nn.Dropout(0.2), nn.Linear(1024, self.num_classes))
        self.register_buffer("class_weights", torch.ones(self.num_classes), persistent=False)

    def extract_features(self, x224, tissues):
        feat = self.uni_encoder(x224)
        feat = self.feat_norm(feat)
        t = self.tissue_embedder(tissues)
        gamma, beta = self.tissue_film(t)
        feat = feat * (1 + gamma) + beta
        return feat
    def forward(self, x224, tissues):
        return self.head(self.extract_features(x224, tissues))

def load_model_from_ckpt(ckpt_path:str, device:str="cuda"):
    obj=torch.load(ckpt_path, map_location="cpu")
    hp=obj.get("hyper_parameters",{}) or {}
    class_names=hp.get("class_names", None); num_tissues=int(hp.get("num_tissues",8))
    if class_names is None: raise RuntimeError("Checkpoint missing hyper_parameters.class_names.")
    # Ensure tissue order matches what the model expects
    expected = ["breast","colon","heart","kidney","liver","lung","ovary","pancreas"]
    assert set(TISSUE_TO_IDX.keys())==set(expected) and all(TISSUE_TO_IDX[k]==i for i,k in enumerate(expected)), \
        f"TISSUE_TO_IDX must match the sorted training order: {expected}"
    model=AOnlyModule(num_classes=len(class_names), class_names=class_names, num_tissues=num_tissues)
    missing,unexpected=model.load_state_dict(obj.get("state_dict", obj), strict=False)
    if missing:    print(f"[ckpt] missing({len(missing)}): {missing[:6]} ...")
    if unexpected: print(f"[ckpt] unexpected({len(unexpected)}): {unexpected[:6]} ...")
    return model.to(device).eval(), list(class_names), num_tissues

# ---------------- h5 & preprocess ----------------
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
def _stack_norm(imgs, device):
    x=torch.stack([torch.from_numpy(im).permute(2,0,1) for im in imgs],0).to(device=device,dtype=torch.float32).div_(255.)
    return (x - MEAN.to(device)) / STD.to(device)

class H5Reader:
    def __init__(self, p): self.p=p; self.f=None; self.merged=False; self.k2i=None
    def _open(self):
        if self.f is None:
            f=h5py.File(self.p,"r"); self.f=f
            if "images" in f and "keys" in f:
                try: keys=f["keys"].asstr()[()]
                except Exception: keys=[k.decode() for k in f["keys"][()]]
                self.k2i={str(k):i for i,k in enumerate(keys)}; self.merged=True
    def get(self,k):
        self._open(); k=str(k)
        if self.merged:
            idx=self.k2i.get(k); 
            if idx is None: raise KeyError(f"{k} not in merged h5")
            arr=self.f["images"][idx][()]
        else:
            if k not in self.f: raise KeyError(k)
            arr=self.f[k][()]
        if arr.ndim==2: arr=np.stack([arr]*3,-1)
        if arr.ndim==3 and arr.shape[-1]==4: arr=arr[...,:3]
        return np.clip(arr,0,255).astype(np.uint8)

# ---------------- utils (print/save) ----------------
def _print_counts(title:str, d:Dict[str,int]):
    items=sorted(d.items(), key=lambda kv:(-kv[1], kv[0])); print(f"  - {title} ({len(items)})")
    for k,v in items[:40]: print(f"    {k:>28s} : {v}")
    if len(items)>40: print(f"    ... ({len(items)-40} more)")

def _perclass_and_fig(y_true, y_pred, names, out_prefix):
    K=len(names); f1=f1_score(y_true,y_pred,labels=list(range(K)),average=None,zero_division=0)
    acc=[float((y_pred[y_true==c]==c).mean()) if np.any(y_true==c) else 0.0 for c in range(K)]
    pd.DataFrame({"class":names,"f1":f1}).to_csv(out_prefix+"_per_class_f1.csv",index=False)
    pd.DataFrame({"class":names,"acc":acc}).to_csv(out_prefix+"_per_class_acc.csv",index=False)
    cm=confusion_matrix(y_true,y_pred,labels=list(range(K)),normalize='true')
    plt.figure(figsize=(max(10,K*0.7),8)); sns.heatmap(cm,annot=True,fmt=".2%",cmap="Blues",xticklabels=names,yticklabels=names)
    plt.title("Confusion Matrix (aligned)"); plt.tight_layout(); plt.savefig(out_prefix+"_confusion_matrix.png",dpi=200); plt.close()

# ---------------- build eval dfs ----------------
def build_xenium_exact_df(ann_json:str, train_names:List[str]):
    print(f"  - Reading annotations from {ann_json}")
    df=pd.read_json(ann_json,orient="index").reset_index().rename(columns={"index":"cell_id"})
    for k in ["class_id","label","cell_type","type","annotation"]:
        if k in df.columns: lbl=k; break
    else: raise KeyError("annotations.json is missing the label column")
    df["orig"]=df[lbl].astype(str).apply(_canon_label)
    cnt_before=df["orig"].value_counts().to_dict()

    # Drop explicit classes that were never trained on
    df=df[~df["orig"].isin(EXACT_REMOVE)].copy()
    # Map into the training label space
    df["aligned"]=df["orig"].map(EXACT_TO_TRAIN_ID).fillna(df["orig"])

    # Keep only labels that still exist in the training classes
    present=set(train_names)
    miss_mask=~df["aligned"].isin(present)
    if miss_mask.any():
        dropped=df.loc[miss_mask,"aligned"].value_counts().to_dict()
        _print_counts("Dropped (not in training label space)", dropped)
        df=df[~miss_mask].copy()

    used=sorted(df["aligned"].unique().tolist())
    c2i={n:i for i,n in enumerate(used)}; df["label_used"]=df["aligned"].map(c2i).astype(int)

    cnt_after=df["aligned"].value_counts().to_dict()
    print(f"  - Samples={len(df)} | classes before mapping={len(cnt_before)} | after mapping={len(used)}")
    _print_counts("Label counts before mapping", cnt_before); _print_counts("Evaluation classes (post-mapping)", cnt_after)

    # Column alignment diagnostics
    miss_in_train=[u for u in used if u not in present]
    assert not miss_in_train, f"Missing classes relative to training: {miss_in_train}"
    used_index_in_train=[train_names.index(n) for n in used]
    print("  - Column alignment (used -> train index):")
    for n,i in zip(used, used_index_in_train):
        print(f"      {n:28s} -> train_col {i}")
    return df[["cell_id","label_used","aligned"]], used, used_index_in_train

# ---------------- inference per holdout ----------------
@torch.no_grad()
def infer_one_holdout(model:AOnlyModule, train_names:List[str], holdout_name:str, root:str, mode:str, tissue:str, out_dir:str, device:str="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    ann_json=os.path.join(root,"annotations.json")
    h5_path=os.path.join(root,"patches_224x224.h5")
    if not os.path.exists(ann_json): raise FileNotFoundError(ann_json)
    if not os.path.exists(h5_path):  raise FileNotFoundError(h5_path)

    if mode!="xenium_exact":
        raise ValueError(f"Unsupported mode '{mode}'. Only 'xenium_exact' is available.")
    df, used_classes, used_index_in_train = build_xenium_exact_df(ann_json, train_names)

    with open(os.path.join(out_dir,"class_names_used.json"),"w") as f: json.dump(used_classes, f, indent=2)

    reader=H5Reader(h5_path)
    organ_id=TISSUE_TO_IDX.get(tissue,0); organ_vec=torch.full((1,), organ_id, dtype=torch.long, device=device)
    B=1024; feats_all=[]; logits_all=[]; probs_used_all=[]; y_all=[]; ids=[]
    for s in tqdm(range(0,len(df),B), desc=f"[{holdout_name}] infer"):
        part=df.iloc[s:s+B]
        imgs=[reader.get(cid) for cid in part["cell_id"].tolist()]
        x=_stack_norm(imgs, device=torch.device(device))
        t=organ_vec.expand(x.shape[0])
        with torch.autocast(device_type=("cuda" if device=="cuda" else "cpu"), dtype=torch.bfloat16, enabled=(device=="cuda")):
            feats=model.extract_features(x,t); logits=model.head(feats); p_train=torch.softmax(logits,dim=1)
            idx=torch.tensor(used_index_in_train, dtype=torch.long, device=p_train.device)
            p_used=p_train.index_select(1, idx)
        feats_all.append(feats.float().cpu()); logits_all.append(logits.float().cpu())
        probs_used_all.append(p_used.float().cpu()); y_all.append(torch.tensor(part["label_used"].values, dtype=torch.long))
        ids += part["cell_id"].tolist()

    feats=np.concatenate([t.numpy() for t in feats_all],0)
    logits=np.concatenate([t.numpy() for t in logits_all],0)
    p_used=np.concatenate([t.numpy() for t in probs_used_all],0)
    y=np.concatenate([t.numpy() for t in y_all],0)
    yhat=p_used.argmax(1)

    np.save(os.path.join(out_dir,"embeddings.npy"), feats.astype(np.float32))
    np.save(os.path.join(out_dir,"probe.npy"),       feats.astype(np.float32))
    np.save(os.path.join(out_dir,"logits.npy"),      logits.astype(np.float32))
    np.save(os.path.join(out_dir,"probs_used.npy"),  p_used.astype(np.float32))
    np.save(os.path.join(out_dir,"y_true.npy"),      y.astype(np.int64))
    np.save(os.path.join(out_dir,"y_pred.npy"),      yhat.astype(np.int64))
    np.save(os.path.join(out_dir,"cell_ids.npy"),    np.array(ids, dtype=object))

    acc=float(accuracy_score(y,yhat)); f1m=float(f1_score(y,yhat,average="macro",zero_division=0))
    with open(os.path.join(out_dir,"metrics.json"),"w") as f:
        json.dump({"n_samples":int(len(y)),"classes_eval":used_classes,"overall_acc":acc,"macro_f1":f1m}, f, indent=2)
    _perclass_and_fig(y,yhat,used_classes, os.path.join(out_dir,"aligned"))
    print(f"  -> saved: {out_dir} | acc={acc:.4f} f1m={f1m:.4f}")

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Path-A inference on holdout datasets.")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="Path to Path-A ckpt")
    parser.add_argument("--holdouts", type=str, default=str(DEFAULT_HOLDOUTS_FILE),
                        help="JSON file describing holdouts (root/mode/tissue).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                        help="Directory to store inference outputs.")
    parser.add_argument("--select", nargs="+", default=None, help="Optional subset of holdout names to run.")
    parser.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    holdouts = load_holdouts_config(Path(args.holdouts))
    selected = args.select or list(holdouts.keys())

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(args.gpu_index)
        print(f"[CUDA] {torch.cuda.get_device_name(torch.cuda.current_device())}")

    ckpt_path = str(Path(args.ckpt).expanduser().resolve())
    model, train_names, _ = load_model_from_ckpt(ckpt_path, device=device)
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "train_class_names.json", "w", encoding="utf-8") as f:
        json.dump(train_names, f, indent=2)

    for name in selected:
        if name not in holdouts:
            raise KeyError(f"Holdout '{name}' not found in {args.holdouts}")
        info = holdouts[name]
        out_dir = out_root / name
        infer_one_holdout(model, train_names, name, info["root"], info["mode"],
                          info["tissue"], str(out_dir), device=device)

if __name__ == "__main__":
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass
    main()
