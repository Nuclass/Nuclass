# -*- coding: utf-8 -*-
import os, re, json, warnings, argparse, random
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing as mp

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from Nuclass.utils.torchvision_compat import ensure_torchvision_nms

ensure_torchvision_nms()
from torchvision import transforms
import timm

warnings.filterwarnings("ignore", message=".*No device id is provided.*", category=UserWarning)

try:
    from Nuclass.train.label_mappings import NEW_LABEL_MAPPING, CLASSES_TO_REMOVE
    from Nuclass.train.dataset_defaults import DEFAULT_DATASET_FOLDERS
    from Nuclass.train.config_utils import resolve_data_dirs
except ModuleNotFoundError:
    from label_mappings import NEW_LABEL_MAPPING, CLASSES_TO_REMOVE  # type: ignore
    from dataset_defaults import DEFAULT_DATASET_FOLDERS  # type: ignore
    from config_utils import resolve_data_dirs  # type: ignore

DEFAULT_RESULTS_DIR = Path("results") / "figure_reproduce" / "film_concat_ablation"

# ----- Dataset helpers -----
def get_data_dirs(explicit: list[str] | None) -> list[str]:
    if explicit:
        return [os.path.abspath(p) for p in explicit]
    return resolve_data_dirs(DEFAULT_DATASET_FOLDERS)

# ----- Utilities -----
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def assert_cuda_arch_compat():
    if not torch.cuda.is_available(): return
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 9:
        m = re.match(r"^(\d+)\.(\d+)", torch.__version__)
        ver = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        if ver < (2,1):
            raise SystemExit(f"[ERROR] Detected sm_{major}{minor} GPU but PyTorch {torch.__version__} lacks sm_90. Install PyTorch>=2.1 CUDA12.")

def build_histopath_transforms():
    mean, std = (0.485,0.456,0.406),(0.229,0.224,0.225)
    val_tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean,std)])
    return val_tf

# ----- Data -----
class AOnlyDataset(Dataset):
    def __init__(self, records, h5_path_224, transform_224):
        self.records = records; self.h5_path = h5_path_224; self.transform_224=transform_224
        self.h5_file=None; self._merged=False; self._key_to_idx=None
    def __len__(self): return len(self.records)
    def _open(self):
        if self.h5_file is None:
            f=h5py.File(self.h5_path,"r"); self.h5_file=f
            if "images" in f and "keys" in f:
                self._merged=True
                keys=[(k.decode() if isinstance(k,(bytes,np.bytes_)) else str(k)) for k in f["keys"][()]]
                self._key_to_idx={k:i for i,k in enumerate(keys)}
    def __getitem__(self, idx):
        self._open(); r=self.records[idx]; cid=str(r["cell_id"])
        if self._merged:
            i=self._key_to_idx.get(cid); 
            if i is None: raise KeyError(f"[{self.h5_path}] key not found in merged H5: {cid}")
            img_np=self.h5_file["images"][i][()]
        else:
            if cid not in self.h5_file: raise KeyError(f"[{self.h5_path}] key not found: {cid}")
            img_np=self.h5_file[cid][()]
        if img_np.ndim==2: img_np=np.stack([img_np]*3,axis=-1)
        elif img_np.ndim==3 and img_np.shape[-1]==4: img_np=img_np[...,:3]
        if img_np.dtype!=np.uint8: img_np=np.clip(img_np,0,255).astype(np.uint8)
        if img_np.shape!=(224,224,3): raise ValueError(f"[{self.h5_path}] unexpected shape: {img_np.shape}")
        x224=self.transform_224(img_np)
        y=torch.tensor(r["label"],dtype=torch.long)
        tissue=torch.tensor(r["tissue_id"],dtype=torch.long)
        # Return (x, y, tissue, cell_id) tuples only
        return x224, y, tissue, r["cell_id"]

def safe_collate(batch):
    """
    Safely collate iterable of (x, y, tissue, cell_id):
    - x: [B, 3, 224, 224] tensor
    - y: [B] long
    - tissue: [B] long
    - cell_ids: list[str]
    """
    xs, ys, ts, cids = [], [], [], []
    for item in batch:
        if len(item) == 4:
            x, y, t, cid = item
        elif len(item) == 5:
            x, y, t, _, cid = item  # backward-compatible legacy tuple
        else:
            raise ValueError(f"Unexpected dataset item length: {len(item)}")
        xs.append(x); ys.append(y); ts.append(t); cids.append(str(cid))
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    t = torch.stack(ts, dim=0)
    return x, y, t, cids

class AOnlyDataModule:
    def __init__(self, data_dirs, label_mapping, classes_to_remove, num_workers=16, batch_size=512):
        self.data_dirs=data_dirs; self.label_mapping=label_mapping; self.classes_to_remove=classes_to_remove
        self.transform_224_val=build_histopath_transforms()
        self.num_workers=num_workers; self.batch_size=batch_size
    @staticmethod
    def _norm_tissue(x): return str(x).strip().lower()
    def setup(self):
        records=[]
        for root in self.data_dirs:
            for split in ["train","test"]:
                meta=os.path.join(root,split,"annotations.json")
                if not os.path.exists(meta): print(f"[WARN] {meta} not found"); continue
                df=pd.read_json(meta, orient="index").reset_index().rename(columns={"index":"cell_id"})
                df["cell_id"]=df["cell_id"].astype(str)
                if "tissue_type" not in df.columns: raise KeyError(f"[ERROR] {meta} missing 'tissue_type'")
                df["tissue_type"]=df["tissue_type"].apply(self._norm_tissue)
                df["split"]=split; df["data_dir"]=root
                records.extend(df.to_dict("records"))
        self.tissue_names=sorted({r["tissue_type"] for r in records})
        self.tissue_to_idx={n:i for i,n in enumerate(self.tissue_names)}
        self.num_tissues = len(self.tissue_names)
        filtered=[r for r in records if r.get("class_id") not in self.classes_to_remove]
        mapped=[]
        for r in filtered:
            final=self.label_mapping.get(r.get("class_id"))
            if final: r["final_class"]=final; mapped.append(r)
        self.class_names=sorted({r["final_class"] for r in mapped})
        self.class_to_idx={n:i for i,n in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        for r in mapped:
            r["label"]=self.class_to_idx[r["final_class"]]; r["tissue_id"]=self.tissue_to_idx[r["tissue_type"]]
        self.val_sets=[]
        for root in self.data_dirs:
            va=[r for r in mapped if r["split"]=="test" and r["data_dir"]==root]
            if va:
                self.val_sets.append(AOnlyDataset(va, os.path.join(root,"test","patches_224x224.h5"), self.transform_224_val))
    def make_subset_val_loader_for_root(self, target_root, subset_size, seed=42):
        ds=None
        for d in self.val_sets:
            if target_root in d.h5_path: ds=d; break
        if ds is None: raise RuntimeError(f"Could not locate dataset for {target_root}")
        N=len(ds); rng=np.random.default_rng(seed)
        idx=np.arange(N) if (subset_size is None or subset_size<=0 or subset_size>=N) else rng.choice(N, size=subset_size, replace=False)
        subset=Subset(ds, idx.tolist())
        ctx=mp.get_context("spawn")
        loader=DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
            multiprocessing_context=ctx,
            collate_fn=safe_collate,   # custom collate to keep IDs
        )
        return loader, ds

# ----- Models -----
class FiLM(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super().__init__()
        hidden=max(64, in_dim*2)
        self.net=nn.Sequential(nn.Linear(in_dim,hidden), nn.SiLU(), nn.Linear(hidden, 2*feat_dim))
    def forward(self, t): gb=self.net(t); return gb.chunk(2,dim=-1)

def build_uni2h_encoder():
    uni_cfg={'img_size':224,'patch_size':14,'depth':24,'num_heads':24,'init_values':1e-5,
             'embed_dim':1536,'mlp_ratio':2.66667*2,'num_classes':0,'no_embed_class':True,
             'mlp_layer': timm.layers.SwiGLUPacked,'act_layer': torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True}
    enc=timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)
    if hasattr(enc,"set_grad_checkpointing"): enc.set_grad_checkpointing(True)
    return enc

class FiLMModule(nn.Module):
    def __init__(self, num_classes, num_tissues, tissue_dim=64):
        super().__init__()
        self.uni_encoder=build_uni2h_encoder(); C=self.uni_encoder.embed_dim
        self.tissue_embedder=nn.Embedding(num_tissues, tissue_dim)
        self.tissue_film=FiLM(tissue_dim, C)
        self.feat_norm=nn.LayerNorm(C)
        self.head=nn.Sequential(nn.Dropout(0.2), nn.Linear(C,1024), nn.SiLU(), nn.Dropout(0.2), nn.Linear(1024,num_classes))
    @torch.no_grad()
    def extract_features(self, x, tissues):
        f=self.uni_encoder(x); f=self.feat_norm(f)
        gamma,beta=self.tissue_film(self.tissue_embedder(tissues)); return f*(1+gamma)+beta
    @torch.no_grad()
    def forward(self, x, tissues): return self.head(self.extract_features(x,tissues))

class ConcatModule(nn.Module):
    """Two-layer MLP head: [1536 + tdim] -> 1024 -> num_classes."""
    def __init__(self, num_classes, num_tissues, tissue_dim=32):
        super().__init__()
        self.uni_encoder=build_uni2h_encoder(); C=self.uni_encoder.embed_dim
        self.tissue_embedder=nn.Embedding(num_tissues, tissue_dim)
        self.feat_norm=nn.LayerNorm(C)
        self.head=nn.Sequential(nn.Dropout(0.2), nn.Linear(C+tissue_dim,1024), nn.SiLU(), nn.Dropout(0.2), nn.Linear(1024,num_classes))
    @torch.no_grad()
    def extract_features(self, x, tissues):
        f=self.uni_encoder(x); f=self.feat_norm(f); t=self.tissue_embedder(tissues); return torch.cat([f,t],dim=-1)
    @torch.no_grad()
    def forward(self, x, tissues): return self.head(self.extract_features(x,tissues))

class ConcatModuleSingle(nn.Module):
    """Single linear head: [1536 + tdim] -> num_classes."""
    def __init__(self, num_classes, num_tissues, tissue_dim=32):
        super().__init__()
        self.uni_encoder=build_uni2h_encoder(); C=self.uni_encoder.embed_dim
        self.tissue_embedder=nn.Embedding(num_tissues, tissue_dim)
        self.feat_norm=nn.LayerNorm(C)
        self.head=nn.Linear(C + tissue_dim, num_classes)
    @torch.no_grad()
    def extract_features(self, x, tissues):
        f=self.uni_encoder(x); f=self.feat_norm(f); t=self.tissue_embedder(tissues); return torch.cat([f,t],dim=-1)
    @torch.no_grad()
    def forward(self, x, tissues): return self.head(self.extract_features(x,tissues))

# ----- Checkpoint parsing/remap/loading -----
def inspect_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    hp = ckpt.get("hyper_parameters", {})

    class_names = hp.get("class_names", None)
    num_classes_hp = len(class_names) if isinstance(class_names, (list, tuple)) else None

    model_type = "film" if any(k.startswith("tissue_film.") for k in sd.keys()) else "concat"

    uni_dim = 1536
    num_classes = None
    in_dim_first = None
    tissue_dim = None
    num_tissues = None

    # tissue_embedder
    for k, v in sd.items():
        if k.startswith("tissue_embedder.weight"):
            num_tissues = int(v.shape[0])
            tissue_dim = int(v.shape[1])
            break

    norm_variant = "feat_norm" if "feat_norm.weight" in sd else ("uni_token_norm" if "uni_token_norm.weight" in sd else None)

    has_mlp = any(k.startswith("head.1.weight") for k in sd.keys())
    has_path_head = any(k.endswith("path_a_head.weight") for k in sd.keys()) or \
                    ("classifier.weight" in sd) or \
                    ("head.weight" in sd and "head.1.weight" not in sd)
    head_style = "mlp" if has_mlp else ("single" if has_path_head else "mlp")

    if head_style == "mlp":
        head_w_keys = [k for k in sd.keys() if k.startswith("head.") and k.endswith(".weight")]
        if head_w_keys:
            def idx(k):
                m = re.match(r"head\.(\d+)\.weight", k)
                return int(m.group(1)) if m else -1
            head_w_keys.sort(key=idx)
            last_w = sd[head_w_keys[-1]]
            num_classes = int(last_w.shape[0])
            first_w = sd[head_w_keys[0]]
            in_dim_first = int(first_w.shape[1])
    else:
        for base in ["path_a_head", "classifier", "final_classifier", "cls", "head"]:
            w_key = f"{base}.weight"
            if w_key in sd:
                W = sd[w_key]
                num_classes = int(W.shape[0])
                in_dim_first = int(W.shape[1])
                break

    if num_classes is None and "criterion.weight" in sd:
        num_classes = int(sd["criterion.weight"].shape[0])
    if num_classes is None:
        num_classes = num_classes_hp

    if tissue_dim is None and model_type == "concat" and in_dim_first is not None:
        guess = in_dim_first - uni_dim
        if guess > 0: tissue_dim = int(guess)
    if tissue_dim is None:
        tissue_dim = 64 if model_type == "film" else 32

    info = {
        "type": model_type,
        "num_classes": num_classes,
        "num_tissues": num_tissues,
        "tissue_dim": tissue_dim,
        "class_names": class_names,
        "norm_variant": norm_variant,
        "head_style": head_style,
        "state_dict": sd,
    }
    print(f"[ckpt] {os.path.basename(ckpt_path)} -> type={info['type']}, head={info['head_style']}, "
          f"norm={info['norm_variant']}, num_classes={info['num_classes']}, tissue_dim={info['tissue_dim']}")
    return info

def remap_alt_keys(sd, info):
    """Map alternate checkpoint naming conventions onto this model."""
    sd = dict(sd)
    def move(old, new):
        if old in sd:
            sd[new] = sd.pop(old)

    if info["type"] == "concat":
        # Normalize layer naming
        if info.get("norm_variant") == "uni_token_norm":
            move("uni_token_norm.weight", "feat_norm.weight")
            move("uni_token_norm.bias",   "feat_norm.bias")

        # Map single-layer heads to head.weight/bias
        if info.get("head_style") == "single":
            if "head.weight" not in sd:
                for base in ["path_a_head", "classifier", "final_classifier", "cls", "head"]:
                    if f"{base}.weight" in sd:
                        move(f"{base}.weight", "head.weight")
                        if f"{base}.bias" in sd:
                            move(f"{base}.bias", "head.bias")
                        break
    return sd

def sanitize_state_dict_for_inference(sd, model):
    """Drop unused/conflicting keys; keep only weights matching the target state_dict shapes."""
    ignore_exact = {"class_weights"}
    ignore_prefixes = ("criterion.",)

    msd = model.state_dict()
    clean, dropped = {}, []
    for k, v in sd.items():
        if k in ignore_exact or any(k.startswith(p) for p in ignore_prefixes):
            dropped.append(k); continue
        if k in msd and msd[k].shape == v.shape:
            clean[k] = v; continue
        # Tolerate extra prefixes added by trainers
        if k.startswith("model.") and k[6:] in msd and msd[k[6:]].shape == v.shape:
            clean[k[6:]] = v; continue
        if k.startswith("module.") and k[7:] in msd and msd[k[7:]].shape == v.shape:
            clean[k[7:]] = v; continue
        dropped.append(k)
    return clean, dropped

def load_model_from_ckpt(ckpt_path, dm, device):
    info = inspect_ckpt(ckpt_path)

    if info["num_classes"] is None:
        info["num_classes"] = getattr(dm, "num_classes", None) or len(getattr(dm, "class_names", [])) or 1
        print(f"[load] num_classes missing; falling back to {info['num_classes']} for shape construction.")

    # Build the matching structure
    if info["type"] == "film":
        model = FiLMModule(num_classes=info["num_classes"], num_tissues=dm.num_tissues, tissue_dim=info["tissue_dim"])
    else:
        if info.get("head_style") == "single":
            model = ConcatModuleSingle(num_classes=info["num_classes"], num_tissues=dm.num_tissues, tissue_dim=info["tissue_dim"])
        else:
            model = ConcatModule(num_classes=info["num_classes"], num_tissues=dm.num_tissues, tissue_dim=info["tissue_dim"])

    # Remap keys + sanitize before loading
    sd_remap = remap_alt_keys(info["state_dict"], info)
    sd_clean, dropped = sanitize_state_dict_for_inference(sd_remap, model)
    if dropped:
        print(f"[load] dropped {len(dropped)} keys (e.g. {dropped[:5]})")

    model.load_state_dict(sd_clean, strict=True)
    model.eval().to(device)
    return model, info

# ----- Class-alignment helpers -----
def compute_alignment(dm_class_names, info_film, info_concat):
    if info_film["class_names"] is None or info_concat["class_names"] is None:
        target=dm_class_names
        film_keep=list(range(info_film["num_classes"]))
        concat_keep=list(range(info_concat["num_classes"]))
    else:
        inter=set(dm_class_names) & set(info_film["class_names"]) & set(info_concat["class_names"])
        target=[n for n in info_film["class_names"] if n in inter]  # keep FiLM ordering
        film_keep=[info_film["class_names"].index(n) for n in target]
        concat_keep=[info_concat["class_names"].index(n) for n in target]
    name2idx={n:i for i,n in enumerate(target)}
    dm_to_target=np.full(len(dm_class_names), -1, dtype=int)
    for i,n in enumerate(dm_class_names):
        if n in name2idx: dm_to_target[i]=name2idx[n]
    return target, dm_to_target, film_keep, concat_keep

@torch.no_grad()
def run_inference(model, loader, device):
    all_emb, all_logits, all_probs, all_y, all_t, all_ids = [], [], [], [], [], []
    for batch in loader:
        # Loader yields (x, y, t, cids) with explicit cell IDs
        x, y, t, cids = batch
        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        emb = model.extract_features(x, t)
        logits = model.head(emb) if hasattr(model, "head") else model(emb)
        probs = torch.softmax(logits, dim=1)

        all_emb.append(emb.cpu())
        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_y.append(y)
        all_t.append(t.cpu())
        all_ids.extend([str(c) for c in cids])

    return {
        "embeddings": torch.cat(all_emb).numpy(),
        "logits": torch.cat(all_logits).numpy(),
        "probs": torch.cat(all_probs).numpy(),
        "labels": torch.cat(all_y).numpy(),
        "tissues": torch.cat(all_t).numpy(),
        "cell_ids": np.array(all_ids, dtype=object)
    }

def align_for_eval(out, keep_cols, dm_to_target):
    map_arr=dm_to_target
    mask=map_arr[out["labels"]] >= 0
    y_target=map_arr[out["labels"][mask]]
    probs_sel=out["probs"][mask][:, keep_cols]
    probs_sel = probs_sel / np.clip(probs_sel.sum(axis=1, keepdims=True), 1e-12, None)
    preds_sel=probs_sel.argmax(axis=1)
    return {
        "embeddings": out["embeddings"][mask],
        "labels": y_target,
        "tissues": out["tissues"][mask],
        "probs": probs_sel,
        "preds": preds_sel,
        "mask": mask
    }

# ----- Metrics & visualization -----
def safe_auroc(y, prob):
    try: return float(roc_auc_score(y, prob, multi_class='ovr', average='macro'))
    except Exception: return None

def knn_accuracy(emb, labels, k=1, metric='cosine', restrict_cross_tissue=False, tissues=None):
    n=emb.shape[0]
    nn=NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1).fit(emb)
    _, idxs=nn.kneighbors(emb); idxs=idxs[:,1:]
    if restrict_cross_tissue:
        assert tissues is not None
        new=np.zeros_like(idxs)
        for i in range(n):
            cand=[j for j in idxs[i] if tissues[j]!=tissues[i]]
            if len(cand)<k: cand=(cand+[idxs[i,-1]]*k)[:k]
            new[i]=np.array(cand[:k])
        idxs=new
    pred=np.apply_along_axis(lambda row: np.bincount(labels[row], minlength=labels.max()+1).argmax(),1,idxs)
    return float((pred==labels).mean())

def tissue_identifiability(emb, tissues, k=5, metric='cosine'):
    nn=NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1).fit(emb)
    _, idxs=nn.kneighbors(emb); idxs=idxs[:,1:]
    pred=np.apply_along_axis(lambda row: np.bincount(tissues[row], minlength=tissues.max()+1).argmax(),1,idxs)
    return float((pred==tissues).mean())

def tissue_leakage_index(emb, labels, tissues):
    emb=emb.astype(np.float64)
    classes=np.unique(labels); tset=np.unique(tissues)
    # Between-class center dispersion
    cls_centers=np.vstack([emb[labels==c].mean(axis=0) for c in classes])
    inter=np.mean(np.linalg.norm(cls_centers[:,None,:]-cls_centers[None,:,:], axis=2))
    intra=[]
    for c in classes:
        centers=[]
        for t in tset:
            m=(labels==c)&(tissues==t)
            if m.sum()>=5: centers.append(emb[m].mean(axis=0))
        if len(centers)>=2:
            centers=np.vstack(centers)
            pd=np.linalg.norm(centers[:,None,:]-centers[None,:,:], axis=2)
            tri=pd[np.triu_indices_from(pd,k=1)]
            if len(tri)>0: intra.append(tri.mean())
    intra=np.mean(intra) if len(intra)>0 else 0.0
    return float(intra/(inter+1e-8))

def reliability_diagram(probs, labels, n_bins=10):
    conf=probs.max(axis=1); preds=probs.argmax(axis=1)
    correct=(preds==labels).astype(np.float32)
    bins=np.linspace(0.0,1.0,n_bins+1); ece=0.0
    xs,accs,confs,counts=[],[],[],[]
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        m=(conf>=lo)&((conf<hi) if i<n_bins-1 else (conf<=hi))
        xs.append((lo+hi)/2.0)
        if m.sum()==0: accs.append(np.nan); confs.append(np.nan); counts.append(0); continue
        accs.append(correct[m].mean()); confs.append(conf[m].mean()); counts.append(int(m.sum()))
        ece += (m.mean())*abs(accs[-1]-confs[-1])
    return np.array(xs),np.array(accs),np.array(confs),np.array(counts),float(ece)

def embed_2d(emb, metric='cosine', random_state=42):
    try:
        import umap.umap_ as umap
        Z=umap.UMAP(n_neighbors=30,min_dist=0.1,metric=metric,random_state=random_state).fit_transform(emb); return Z,"UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        Z=TSNE(n_components=2,perplexity=30,learning_rate='auto',init='pca',metric=metric,random_state=random_state).fit_transform(emb)
        return Z,"t-SNE"

def plot_scatter(ax, Z, color, title):
    palette=sns.color_palette(n_colors=int(color.max()+1))
    for c in np.unique(color):
        m=(color==c); ax.scatter(Z[m,0],Z[m,1],s=5,alpha=0.8,label=str(c),color=palette[int(c)%len(palette)])
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.legend(markerscale=3,fontsize=8,frameon=False,ncol=4)

# ----- Main -----
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt_film", required=True); ap.add_argument("--ckpt_concat", required=True)
    ap.add_argument("--target_root", required=True); ap.add_argument("--subset_size", type=int, default=7000)
    ap.add_argument("--num_workers", type=int, default=16); ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--data-dirs", nargs="+", default=None, help="Override dataset directories; defaults to NUCLASS_DATA_* env vars.")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--gpu_index", type=int, default=0)
    args=ap.parse_args()

    out_dir = Path(args.results_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.device=="cuda" and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index); assert_cuda_arch_compat()
    device=torch.device("cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    torch.set_float32_matmul_precision('medium')

    # Data
    data_dirs = get_data_dirs(args.data_dirs)
    dm=AOnlyDataModule(data_dirs, NEW_LABEL_MAPPING, CLASSES_TO_REMOVE,
                       num_workers=args.num_workers, batch_size=args.batch_size)
    dm.setup()
    loader,_=dm.make_subset_val_loader_for_root(args.target_root, args.subset_size, seed=args.seed)
    print(f"[INFO] subset size={len(loader.dataset)}; num_classes={len(dm.class_names)}; num_tissues={len(dm.tissue_names)}")

    # Parse checkpoints and find class intersection
    film_info=inspect_ckpt(args.ckpt_film)
    concat_info=inspect_ckpt(args.ckpt_concat)
    target_names, dm_to_target, film_keep, concat_keep = compute_alignment(dm.class_names, film_info, concat_info)
    K=len(target_names)
    print(f"[ALIGN] intersected classes={K} -> {target_names}")

    # Load models with matching architectures
    film_model,_   = load_model_from_ckpt(args.ckpt_film, dm, device)
    concat_model,_ = load_model_from_ckpt(args.ckpt_concat, dm, device)

    # Inference
    film_raw  = run_inference(film_model, loader, device)
    concat_raw= run_inference(concat_model, loader, device)

    # Align outputs to the shared class space
    film  = align_for_eval(film_raw,  film_keep,  dm_to_target)
    concat= align_for_eval(concat_raw, concat_keep, dm_to_target)
    assert film["labels"].shape[0]==concat["labels"].shape[0], "Mismatched sample counts after alignment"
    N_eff=film["labels"].shape[0]
    print(f"[ALIGN] effective samples after alignment: {N_eff}")

    # ---- Metrics ----
    def summarize(out):
        y,pred,prob=out["labels"], out["preds"], out["probs"]
        acc=float(accuracy_score(y,pred))
        f1m=float(f1_score(y,pred,average="macro",zero_division=0))
        auroc=safe_auroc(y,prob)
        return acc,f1m,auroc

    acc_f,f1_f,auroc_f = summarize(film)
    acc_c,f1_c,auroc_c = summarize(concat)

    # Paired bootstrap delta with quantile-based CI
    def bootstrap_ci_diff(metric_fn, y, a_pred, b_pred, n_boot=1000, seed=42):
        """
        Paired bootstrap for delta = metric(a) - metric(b)
        Returns (mean, [2.5%, 97.5%] CI)
        """
        rng = np.random.default_rng(seed)
        n = len(y)
        diffs = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)  # sample with replacement
            diffs[b] = metric_fn(y[idx], a_pred[idx]) - metric_fn(y[idx], b_pred[idx])
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        return float(diffs.mean()), (float(lo), float(hi))

    mean_d_acc,(ci_l_acc,ci_u_acc) = bootstrap_ci_diff(
        lambda y,p: accuracy_score(y,p),
        film["labels"], film["preds"], concat["preds"], n_boot=1000, seed=args.seed
    )
    mean_d_f1,(ci_l_f1,ci_u_f1) = bootstrap_ci_diff(
        lambda y,p: f1_score(y,p,average="macro",zero_division=0),
        film["labels"], film["preds"], concat["preds"], n_boot=1000, seed=args.seed
    )

    # Embedding separability
    from sklearn.preprocessing import normalize
    E_f=normalize(film["embeddings"]); E_c=normalize(concat["embeddings"])
    sil_f=silhouette_score(E_f, film["labels"], metric='cosine'); sil_c=silhouette_score(E_c, concat["labels"], metric='cosine')
    ch_f=calinski_harabasz_score(E_f, film["labels"]); ch_c=calinski_harabasz_score(E_c, concat["labels"])
    db_f=davies_bouldin_score(E_f, film["labels"]); db_c=davies_bouldin_score(E_c, concat["labels"])

    # kNN
    knn_same_f  = knn_accuracy(E_f, film["labels"], k=1, metric='cosine', restrict_cross_tissue=False, tissues=film["tissues"])
    knn_same_c  = knn_accuracy(E_c, concat["labels"], k=1, metric='cosine', restrict_cross_tissue=False, tissues=concat["tissues"])
    knn_cross_f = knn_accuracy(E_f, film["labels"], k=1, metric='cosine', restrict_cross_tissue=True, tissues=film["tissues"])
    knn_cross_c = knn_accuracy(E_c, concat["labels"], k=1, metric='cosine', restrict_cross_tissue=True, tissues=concat["tissues"])
    tir_f       = tissue_identifiability(E_f, film["tissues"], k=5, metric='cosine')
    tir_c       = tissue_identifiability(E_c, concat["tissues"], k=5, metric='cosine')
    tli_f       = tissue_leakage_index(E_f, film["labels"], film["tissues"])
    tli_c       = tissue_leakage_index(E_c, concat["labels"], concat["tissues"])

    # 2D visualization
    Zf,methF=embed_2d(E_f, metric='cosine', random_state=args.seed)
    Zc,methC=embed_2d(E_c, metric='cosine', random_state=args.seed)

    # Confusion and calibration
    cm_f=confusion_matrix(film["labels"], film["preds"], labels=np.arange(K), normalize="true")
    cm_c=confusion_matrix(concat["labels"], concat["preds"], labels=np.arange(K), normalize="true")
    cm_diff=cm_f - cm_c
    xs_f,accs_f,confs_f,counts_f,ece_f = reliability_diagram(film["probs"], film["labels"], n_bins=10)
    xs_c,accs_c,confs_c,counts_c,ece_c = reliability_diagram(concat["probs"], concat["labels"], n_bins=10)

    # Per-tissue accuracy
    tnames=getattr(dm,"tissue_names",[f"t{i}" for i in np.unique(film["tissues"])])
    per_tissue_acc_f=[]; per_tissue_acc_c=[]
    for tid in np.unique(film["tissues"]):
        m=(film["tissues"]==tid)
        per_tissue_acc_f.append(accuracy_score(film["labels"][m], film["preds"][m]))
        per_tissue_acc_c.append(accuracy_score(concat["labels"][m], concat["preds"][m]))

    # ---- PDF export ----
    pdf_name=f"FiLM_vs_Concat_subset{N_eff}_K{K}_seed{args.seed}.pdf"
    pdf_path=os.path.join(out_dir ,pdf_name)
    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary
        fig,ax=plt.subplots(figsize=(10,5)); ax.axis('off')
        rows=[
            ["Metric","FiLM","Concat","Delta(F-C)"],
            ["ACC", f"{acc_f:.4f}", f"{acc_c:.4f}", f"{acc_f-acc_c:+.4f} (95% CI {ci_l_acc:+.4f},{ci_u_acc:+.4f})"],
            ["F1-macro", f"{f1_f:.4f}", f"{f1_c:.4f}", f"{f1_f-f1_c:+.4f} (95% CI {ci_l_f1:+.4f},{ci_u_f1:+.4f})"],
            ["AUROC-macro", f"{(auroc_f if auroc_f is not None else np.nan):.4f}",
             f"{(auroc_c if auroc_c is not None else np.nan):.4f}",
             f"{(0 if (auroc_f is None or auroc_c is None) else (auroc_f-auroc_c)):+.4f}"],
            ["Silhouette (higher)", f"{sil_f:.4f}", f"{sil_c:.4f}", f"{sil_f-sil_c:+.4f}"],
            ["Calinski-Harabasz (higher)", f"{ch_f:.1f}", f"{ch_c:.1f}", f"{ch_f-ch_c:+.1f}"],
            ["Davies-Bouldin (lower)", f"{db_f:.4f}", f"{db_c:.4f}", f"{db_f-db_c:+.4f}"],
            ["1-NN acc (same tissue, higher)", f"{knn_same_f:.4f}", f"{knn_same_c:.4f}", f"{knn_same_f-knn_same_c:+.4f}"],
            ["1-NN acc (cross tissue, higher)", f"{knn_cross_f:.4f}", f"{knn_cross_c:.4f}", f"{knn_cross_f-knn_cross_c:+.4f}"],
            ["kNN Tissue Identifiability (lower)", f"{tir_f:.4f}", f"{tir_c:.4f}", f"{tir_f-tir_c:+.4f}"],
            ["Tissue Leakage Index (lower)", f"{tli_f:.4f}", f"{tli_c:.4f}", f"{tli_f-tli_c:+.4f}"],
            ["ECE (lower)", f"{ece_f:.4f}", f"{ece_c:.4f}", f"{ece_f-ece_c:+.4f}"],
        ]
        tb=ax.table(cellText=rows, loc='center', cellLoc='center', colWidths=[0.22,0.22,0.22,0.34])
        tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1,1.2)
        ax.set_title(f"FiLM vs Concat on {args.target_root.split('/')[-1]} (N={N_eff}, K={K})", fontsize=14)
        pdf.savefig(fig); plt.close(fig)

        # Page 2-3: 2D embeddings
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        plot_scatter(axes[0], Zf, film["labels"], f"{methF} (FiLM) - by CLASS")
        plot_scatter(axes[1], Zc, concat["labels"], f"{methC} (Concat) - by CLASS")
        pdf.savefig(fig); plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(14,6))
        plot_scatter(axes[0], Zf, film["tissues"], f"{methF} (FiLM) - by TISSUE")
        plot_scatter(axes[1], Zc, concat["tissues"], f"{methC} (Concat) - by TISSUE")
        pdf.savefig(fig); plt.close(fig)

        # Page 4: per-tissue accuracy
        fig,ax=plt.subplots(figsize=(max(8, len(tnames)*0.6),5))
        x=np.arange(len(tnames))
        ax.bar(x-0.2, per_tissue_acc_f, width=0.4, label="FiLM")
        ax.bar(x+0.2, per_tissue_acc_c, width=0.4, label="Concat")
        ax.set_xticks(x); ax.set_xticklabels(tnames, rotation=45, ha='right')
        ax.set_ylim(0,1); ax.set_ylabel("Accuracy"); ax.set_title("Per-tissue Accuracy"); ax.legend()
        pdf.savefig(fig); plt.close(fig)

        # Page 5: Confusion matrices
        fig,axes=plt.subplots(1,3,figsize=(18,6))
        sns.heatmap(cm_f, ax=axes[0], cmap="Blues", vmin=0, vmax=1)
        axes[0].set_title("Confusion (FiLM)"); axes[0].set_xlabel("Pred"); axes[0].set_ylabel("True")
        sns.heatmap(cm_c, ax=axes[1], cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Confusion (Concat)"); axes[1].set_xlabel("Pred"); axes[1].set_ylabel("True")
        v=np.max(np.abs(cm_diff))
        sns.heatmap(cm_diff, ax=axes[2], cmap="PiYG", center=0, vmin=-v, vmax=v)
        axes[2].set_title("Delta Confusion (FiLM-Concat)"); axes[2].set_xlabel("Pred"); axes[2].set_ylabel("True")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Page 6: Reliability
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for ax,xs,accs,confs,counts,ece,title in [
            (axes[0], xs_f, accs_f, confs_f, counts_f, ece_f, f"FiLM Reliability (ECE={ece_f:.3f})"),
            (axes[1], xs_c, accs_c, confs_c, counts_c, ece_c, f"Concat Reliability (ECE={ece_c:.3f})")
        ]:
            ax.plot([0,1],[0,1],'--',linewidth=1,color='gray')
            ax.plot(xs,accs,marker='o',label='Accuracy per bin')
            ax.plot(xs,confs,marker='s',label='Confidence per bin')
            ax2=ax.twinx(); ax2.bar(xs,counts,width=0.08,alpha=0.3,edgecolor='black')
            ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax2.set_ylabel("Count")
            ax.set_title(title); ax.legend(loc='lower right')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    out_json={
        "subset_size": int(N_eff),
        "K_classes": K,
        "target_class_names": target_names,
        "metrics": {
            "acc": {"film": acc_f, "concat": acc_c, "delta": acc_f-acc_c, "bootstrap_CI":[ci_l_acc, ci_u_acc]},
            "f1_macro": {"film": f1_f, "concat": f1_c, "delta": f1_f-f1_c, "bootstrap_CI":[ci_l_f1, ci_u_f1]},
            "auroc_macro": {"film": auroc_f, "concat": auroc_c, "delta": None if (auroc_f is None or auroc_c is None) else (auroc_f-auroc_c)},
            "silhouette": {"film": float(sil_f), "concat": float(sil_c), "delta": float(sil_f-sil_c)},
            "calinski_harabasz": {"film": float(ch_f), "concat": float(ch_c), "delta": float(ch_f-ch_c)},
            "davies_bouldin": {"film": float(db_f), "concat": float(db_c), "delta": float(db_f-db_c)},
            "knn_same_tissue_acc": {"film": knn_same_f, "concat": knn_same_c, "delta": knn_same_f-knn_same_c},
            "knn_cross_tissue_acc": {"film": knn_cross_f, "concat": knn_cross_c, "delta": knn_cross_f-knn_cross_c},
            "tissue_identifiability": {"film": tir_f, "concat": tir_c, "delta": tir_f-tir_c},
            "tissue_leakage_index": {"film": tli_f, "concat": tli_c, "delta": tli_f-tli_c},
            "ece": {"film": ece_f, "concat": ece_c, "delta": ece_f-ece_c},
        },
        "paths": {"pdf": pdf_path}
    }
    with open(os.path.join(out_dir, os.path.splitext(os.path.basename(pdf_path))[0]+".json"), "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[OK] report saved to: {pdf_path}")

if __name__=="__main__":
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass
    main()
