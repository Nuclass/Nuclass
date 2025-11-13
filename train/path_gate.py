# -*- coding: utf-8 -*-
"""Path-local/global fusion gate (speed-focused, Triton/Inductor optional)."""

import os, json, time, warnings, math, hashlib, pickle, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext

# ====== Environment tweaks (before torch/h5py imports) ======
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "TRUE")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

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

# ===================== CONFIG =====================
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "path_gate.yaml"
CONFIG = None


def prepare_config():
    global CONFIG
    if CONFIG is not None:
        return CONFIG

    config, _ = load_experiment_config(
        default_config_path=DEFAULT_CONFIG_PATH,
        description="Path-local/global fusion gate training",
    )
    config["data_dirs"] = resolve_data_dirs(DEFAULT_DATASET_FOLDERS)
    config["inference_out_dir"] = resolve_results_dir(config["experiment_name"])
    config["checkpoint_dir"] = get_checkpoint_dir(config["experiment_name"])
    config["ckpt_pathA"] = resolve_checkpoint_path(config.pop("ckpt_pathA_rel"))
    config["ckpt_pathB"] = resolve_checkpoint_path(config.pop("ckpt_pathB_rel"))
    CONFIG = config
    return CONFIG

import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ---- fail-safe: fall back to eager when Dynamo/Inductor errors ----
try:
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True
except Exception:
    pass

from Nuclass.utils.torchvision_compat import ensure_torchvision_nms

ensure_torchvision_nms()
import timm, pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoModel, AutoConfig, logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*No device id is provided.*", category=UserWarning)

# ================= Adaptive compile helpers =================
def _has_libcuda() -> bool:
    cand = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib64/libcuda.so",
        "/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/local/cuda/lib64/libcuda.so",
        "/usr/lib/wsl/lib/libcuda.so",
    ]
    return any(os.path.exists(p) for p in cand)

def _decide_backend(cfg: dict) -> str:
    req = str(cfg.get("use_torch_compile", "off")).lower()
    backend = str(cfg.get("compile_backend", "inductor")).lower()
    if req == "off":
        return "off"
    if req in ("inductor", "aot_eager"):
        return req
    if backend == "inductor":
        if shutil.which("gcc") and _has_libcuda():
            return "inductor"
        else:
            print("[compile] Missing libcuda/gcc for Inductor; falling back to 'aot_eager'")
            return "aot_eager"
    return "aot_eager"

def _maybe_compile(module: nn.Module, name: str, cfg: dict) -> nn.Module:
    backend = _decide_backend(cfg)
    if backend == "off":
        print(f"[compile] skip {name} (compile=off)")
        return module
    try:
        print(f"[compile] compiling {name} -> backend={backend}, mode={cfg.get('compile_mode','reduce-overhead')}")
        return torch.compile(module,
                             backend=backend,
                             mode=cfg.get("compile_mode", "reduce-overhead"),
                             fullgraph=False)
    except Exception as e:
        print(f"[compile] {name} failed to compile, falling back to eager: {e.__class__.__name__}: {e}")
        return module

# ================= timm compatibility: SwiGLUPacked / SwiGLU =================
try:
    from timm.layers import SwiGLUPacked as _MLP
except Exception:
    from timm.layers import SwiGLU as _MLP

# ==================== IO & helpers ====================
def log_stage(msg, t0): print(f"[{time.perf_counter()-t0:6.2f}s] {msg}", flush=True)

def _h5_open_read(path: str):
    cfg = prepare_config()
    kwargs = dict(mode="r", libver="latest",
                  swmr=bool(cfg.get("h5_open_swmr", False)),
                  rdcc_nbytes=int(cfg.get("h5_rdcc_nbytes", 256*1024*1024)),
                  rdcc_nslots=int(cfg.get("h5_rdcc_nslots", 1_000_003)),
                  rdcc_w0=float(cfg.get("h5_rdcc_w0", 0.75)))
    try: return h5py.File(path, **kwargs)
    except TypeError: return h5py.File(path, "r")

def _file_sig(path: str) -> str:
    st = os.stat(path); return hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()

def _ensure_cache_dir() -> str:
    cfg = prepare_config()
    d = os.path.expanduser(cfg.get("h5_index_cache_dir", "~/.cache/pathb_h5"))
    os.makedirs(d, exist_ok=True); return d

def _load_or_build_k2i(h5_path: str) -> Optional[Dict[str,int]]:
    if not os.path.exists(h5_path): return None
    cfg = prepare_config()
    use_disk = bool(cfg.get("use_h5_index_cache", False))
    cache_dir=_ensure_cache_dir(); cf=os.path.join(cache_dir, f"{_file_sig(h5_path)}.k2i.pkl")
    if use_disk and os.path.exists(cf):
        try: return pickle.load(open(cf,"rb"))
        except Exception: pass
    t0=time.perf_counter()
    with _h5_open_read(h5_path) as f:
        if "images" in f and "keys" in f:
            ds=f["keys"]
            try: keys=ds.asstr()[()]
            except Exception:
                raw=ds[()][:]
                keys=[k.decode() if isinstance(k,(bytes,np.bytes_)) else str(k) for k in raw]
            k2i={str(k):int(i) for i,k in enumerate(keys)}
        else: k2i=None
    log_stage(f"build k2i for {os.path.basename(h5_path)}", t0)
    if use_disk and k2i is not None:
        try: pickle.dump(k2i, open(cf,"wb"), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception: pass
    return k2i

def _maybe_ctx_aligns_with_224(h5_ctx: str, k2i_224: Optional[Dict[str,int]], samples:int=4)->bool:
    if k2i_224 is None or not os.path.exists(h5_ctx): return False
    try:
        with _h5_open_read(h5_ctx) as f:
            if "keys" not in f: return False
            ds=f["keys"]; N=int(ds.shape[0])
            idxs=[0, N//3, (2*N)//3, N-1] if N>3 else list(range(N))
            hit=0
            for i in idxs[:max(1,samples)]:
                try: k=ds.asstr()[i]
                except Exception:
                    raw=ds[i]; k=raw.decode() if isinstance(raw,(bytes,np.bytes_)) else str(raw)
                if k2i_224.get(str(k), -9999)==i: hit+=1
            return hit >= max(1, len(idxs[:samples])-1)
    except Exception: return False

# ==================== Dataset / DataModule ====================
class PathBDataset(Dataset):
    def __init__(self, records, h5_224, h5_ctx, ctx_target_size, k2i224=None, k2iCTX=None, ctx_read_mode="auto"):
        self.records=records
        self.h5_224_path=h5_224; self.h5_ctx_path=h5_ctx
        self.ctx_target=int(ctx_target_size); self.ctx_read_mode=str(ctx_read_mode).lower()
        self.h5_224=None; self.h5_ctx=None
        self.k2i224=k2i224; self.k2iCTX=k2iCTX
        self.merged224=None; self.mergedCTX=None
        self.ctx_H=1024; self.ctx_W=1024
    def __len__(self): return len(self.records)
    def _open(self):
        if self.h5_224 is None:
            f=_h5_open_read(self.h5_224_path); self.h5_224=f; self.merged224=("images" in f and "keys" in f)
        if self.h5_ctx is None:
            f2=_h5_open_read(self.h5_ctx_path); self.h5_ctx=f2; self.mergedCTX=("images" in f2 and "keys" in f2)
            if self.mergedCTX:
                ds=f2["images"]; self.ctx_H=int(ds.shape[1]); self.ctx_W=int(ds.shape[2])
    @staticmethod
    def _get224(f224, merged, k2i, key)->np.ndarray:
        if merged:
            if k2i is None: raise KeyError("merged224 requires k2i")
            i=k2i.get(key)
            if i is None: raise KeyError(f"[224] missing {key}")
            arr=f224["images"][i][()]
        else:
            if key not in f224: raise KeyError(f"[224] missing {key}")
            arr=f224[key][()]
        if arr.ndim==2: arr=np.stack([arr]*3,-1)
        if arr.ndim==3 and arr.shape[-1]==4: arr=arr[...,:3]
        if arr.dtype!=np.uint8: arr=np.clip(arr,0,255).astype(np.uint8)
        if arr.shape!=(224,224,3): raise ValueError(f"[224] shape bad {arr.shape}")
        return arr
    def _post_ctx(self, arr):
        if arr.ndim==2: arr=np.stack([arr]*3,-1)
        if arr.ndim==3 and arr.shape[-1]==4: arr=arr[...,:3]
        if arr.dtype!=np.uint8: arr=np.clip(arr,0,255).astype(np.uint8)
        return arr
    def _cpu_resize(self, arr):
        if (arr.shape[0], arr.shape[1]) == (self.ctx_target, self.ctx_target):
            return arr
        t=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
        t=F.interpolate(t, size=(self.ctx_target, self.ctx_target), mode='bilinear', align_corners=False)
        return t[0].permute(1,2,0).byte().numpy()

    def _read_ctx_full(self, ds, i):
        arr = ds[i][()] if i is not None else ds[()]
        arr=self._post_ctx(arr)
        arr=self._cpu_resize(arr)
        return arr

    def _read_ctx_strided(self, ds, i, H, W):
        fh=H//self.ctx_target; fw=W//self.ctx_target
        if fh*self.ctx_target==H and fw*self.ctx_target==W and getattr(ds,"compression",None) is None:
            arr = ds[i, ::fh, ::fw, :] if i is not None else ds[::fh, ::fw, :]
            arr=self._post_ctx(arr)
            return arr
        else:
            return self._read_ctx_full(ds, i)

    def _getCTX_auto(self, key)->np.ndarray:
        f=self.h5_ctx
        if self.mergedCTX:
            if self.k2iCTX is None: raise KeyError("mergedCTX requires k2i")
            i=self.k2iCTX.get(key)
            ds=f["images"]; H=int(ds.shape[1]); W=int(ds.shape[2]); self.ctx_H, self.ctx_W = H, W
            if self.ctx_read_mode=="strided": return self._read_ctx_strided(ds,i,H,W)
            if self.ctx_read_mode=="full": return self._read_ctx_full(ds,i)
            chunks=getattr(ds,"chunks",None); comp=getattr(ds,"compression",None)
            if chunks is None and comp is None:
                try: return self._read_ctx_strided(ds,i,H,W)
                except Exception: return self._read_ctx_full(ds,i)
            else:
                return self._read_ctx_full(ds,i)
        else:
            if key not in f: raise KeyError(f"[CTX] missing {key}")
            ds=f[key]; H=int(ds.shape[0]); W=int(ds.shape[1]); self.ctx_H,self.ctx_W=H,W
            return self._read_ctx_full(ds,None)

    def __getitem__(self, idx):
        self._open()
        r=self.records[idx]; cid=str(r["cell_id"])
        img224=self._get224(self.h5_224, self.merged224, self.k2i224, cid)
        imgctx=self._getCTX_auto(cid)  # resized on CPU to ctx_target
        x224=torch.from_numpy(img224).permute(2,0,1).contiguous()
        xctx=torch.from_numpy(imgctx).permute(2,0,1).contiguous()
        y=torch.tensor(r["label"],dtype=torch.long)
        tissue=torch.tensor(r["tissue_id"],dtype=torch.long)
        return x224, xctx, y, tissue

class PathBDataModule(pl.LightningDataModule):
    def __init__(self, config, label_mapping, classes_to_remove):
        super().__init__()
        self.config=config; self.label_mapping=label_mapping; self.classes_to_remove=classes_to_remove
    @staticmethod
    def _norm_tissue(x): return str(x).strip().lower()
    def _fast_preflight(self, root, split):
        for fn in ["patches_224x224.h5","patches_1024x1024.h5"]:
            p=os.path.join(root,split,fn)
            if not os.path.exists(p): raise FileNotFoundError(p)
            with _h5_open_read(p) as f: _=f.id
        print(f"  preflight ok: {os.path.basename(root)}/{split}", flush=True)
    def setup(self, stage=None):
        t0=time.perf_counter(); print("[setup] reading annotations.json ...", flush=True)
        records=[]
        for root in self.config["data_dirs"]:
            for split in ["train","test"]:
                meta=os.path.join(root,split,"annotations.json")
                if not os.path.exists(meta): print(f"[WARN] {meta} not found"); continue
                df=pd.read_json(meta, orient="index").reset_index().rename(columns={'index':'cell_id'})
                df["cell_id"]=df["cell_id"].astype(str)
                if "tissue_type" not in df.columns: raise KeyError(f"{meta} missing tissue_type")
                df["tissue_type"]=df["tissue_type"].apply(self._norm_tissue)
                df["split"]=split; df["data_dir"]=root
                records.extend(df.to_dict('records'))
        log_stage("annotations loaded", t0)

        self.tissue_names=sorted({r["tissue_type"] for r in records})
        self.tissue_to_idx={n:i for i,n in enumerate(self.tissue_names)}
        self.num_tissues=len(self.tissue_names)

        filt=[r for r in records if r.get("class_id") not in self.classes_to_remove]
        mapped=[]
        for r in filt:
            final=self.label_mapping.get(r.get("class_id"))
            if final: r["final_class"]=final; mapped.append(r)
        self.class_names=sorted({r["final_class"] for r in mapped})
        self.num_classes=len(self.class_names); self.class_to_idx={n:i for i,n in enumerate(self.class_names)}
        for r in mapped:
            r["label"]=self.class_to_idx[r["final_class"]]
            r["tissue_id"]=self.tissue_to_idx[r["tissue_type"]]

        self.train_records=[r for r in mapped if r["split"]=="train"]
        self.val_records=[r for r in mapped if r["split"]=="test"]

        t1=time.perf_counter(); print("[setup] fast preflight ...", flush=True)
        for root in self.config["data_dirs"]:
            for split in ["train","test"]:
                meta=os.path.join(root,split,"annotations.json")
                if not os.path.exists(meta): continue
                self._fast_preflight(root,split)
        log_stage("fast preflight done", t1)

        # build k2i on the main process (memory only)
        t2=time.perf_counter(); self.k2i_cache={}
        for root in self.config["data_dirs"]:
            for split in ["train","test"]:
                p224=os.path.join(root,split,"patches_224x224.h5")
                pCTX=os.path.join(root,split,"patches_1024x1024.h5")
                merged224, mergedCTX=False, False
                try:
                    with _h5_open_read(p224) as f: merged224=("images" in f and "keys" in f)
                except Exception: pass
                try:
                    with _h5_open_read(pCTX) as f: mergedCTX=("images" in f and "keys" in f)
                except Exception: pass
                k2i224=_load_or_build_k2i(p224) if merged224 else None
                if mergedCTX and k2i224 is not None and _maybe_ctx_aligns_with_224(pCTX,k2i224,samples=int(self.config.get("keys_check_samples",4))):
                    k2iCTX=k2i224
                else:
                    k2iCTX=_load_or_build_k2i(pCTX) if mergedCTX else None
                self.k2i_cache[p224]=k2i224; self.k2i_cache[pCTX]=k2iCTX
        log_stage("build/load H5 key indices", t2)

        # dataset construction
        t3=time.perf_counter()
        self.train_sets=[]; self.val_sets=[]
        for root in self.config["data_dirs"]:
            tr=[r for r in self.train_records if r["data_dir"]==root]
            va=[r for r in self.val_records if r["data_dir"]==root]
            h5_224_tr=os.path.join(root,"train","patches_224x224.h5")
            h5_ctx_tr=os.path.join(root,"train","patches_1024x1024.h5")
            h5_224_va=os.path.join(root,"test","patches_224x224.h5")
            h5_ctx_va=os.path.join(root,"test","patches_1024x1024.h5")
            if tr:
                self.train_sets.append(PathBDataset(tr,h5_224_tr,h5_ctx_tr,self.config["ctx_img_size"],
                    k2i224=self.k2i_cache.get(h5_224_tr), k2iCTX=self.k2i_cache.get(h5_ctx_tr),
                    ctx_read_mode=self.config.get("ctx_read_mode","auto")))
            if va:
                self.val_sets.append(PathBDataset(va,h5_224_va,h5_ctx_va,self.config["ctx_img_size"],
                    k2i224=self.k2i_cache.get(h5_224_va), k2iCTX=self.k2i_cache.get(h5_ctx_va),
                    ctx_read_mode=self.config.get("ctx_read_mode","auto")))
        log_stage("datasets built", t3)
        print(f"Classes: {self.num_classes}, Train: {len(self.train_records)}, Val: {len(self.val_records)}", flush=True)

    def _resolve_num_workers(self):
        nw = self.config["num_workers"]
        if isinstance(nw, str) and nw.lower()=="auto":
            try:
                cpu = os.cpu_count() or 8
                return max(8, min(32, cpu-2))
            except Exception:
                return 8
        return int(nw)

    def _make_loader(self, ds_list, shuffle):
        per_device_batch = int(self.config.get("per_device_batch") or max(1, self.config["batch_size"] // max(1, self.config.get("accumulate_grad_batches",1))))
        kwargs=dict(
            batch_size=per_device_batch, shuffle=shuffle,
            num_workers=self._resolve_num_workers(),
            pin_memory=True,
            pin_memory_device="cuda" if torch.cuda.is_available() else "",
            prefetch_factor=int(self.config.get("prefetch_factor",4)),
            persistent_workers=bool(self.config.get("persistent_workers", True)),
            drop_last=True,
        )
        kwargs["multiprocessing_context"]=self.config.get("multiprocessing_context","spawn")
        loader=DataLoader(ConcatDataset(ds_list), **kwargs)
        return loader

    def train_dataloader(self): return self._make_loader(self.train_sets, True)
    def val_dataloader(self):   return self._make_loader(self.val_sets, False)
    def test_dataloader(self):  return self.val_dataloader()

# ==================== PathA / PathB ====================
class FiLM(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super().__init__()
        hidden=max(64, in_dim*2)
        self.net=nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2*feat_dim))
    def forward(self, t): return self.net(t).chunk(2, dim=-1)

class PathA_InferExact(nn.Module):
    """Path-A: UNI2-h + FiLM + head (optionally returns pre-head features)."""
    def __init__(self, num_classes, num_tissues, tdim=64):
        super().__init__()
        uni_cfg={'img_size':224,'patch_size':14,'depth':24,'num_heads':24,'init_values':1e-5,
                 'embed_dim':1536,'mlp_ratio':2.66667*2,'num_classes':0,'no_embed_class':True,
                 'mlp_layer':_MLP,'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True}
        self.uni_encoder=timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)
        self.feat_norm=nn.LayerNorm(self.uni_encoder.embed_dim)
        self.tissue_embedder=nn.Embedding(num_tissues, tdim)
        self.tissue_film=FiLM(tdim, self.uni_encoder.embed_dim)
        self.head=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.uni_encoder.embed_dim, 1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def load_from_ckpt(self, ckpt_path: str):
        sd_raw = torch.load(ckpt_path, map_location="cpu")
        state = sd_raw.get("state_dict", sd_raw)
        keep = {k:v for k,v in state.items() if k.startswith(("uni_encoder.","feat_norm.","tissue_embedder.","tissue_film.","head."))}
        self.load_state_dict(keep, strict=False)

    def forward(self, x224, tissue, return_feat=False):
        feat=self.uni_encoder(x224)                 # [B,1536]
        feat=self.feat_norm(feat)
        t=self.tissue_embedder(tissue)
        g,b=self.tissue_film(t)
        feat=feat*(1+g)+b                            # pre-head features
        logits=self.head(feat)                      # [B,C]
        return (logits, feat) if return_feat else logits

class PathB_InferExact(nn.Module):
    """Path-B: UNI(224) + DINO(ctx) -> proj -> concat -> head_B"""
    def __init__(self, num_classes, dino_model, use_pooler=True, ctx_microbatch=16):
        super().__init__()
        self.use_pooler=use_pooler
        self.ctx_microbatch=int(ctx_microbatch)

        uni_cfg={'img_size':224,'patch_size':14,'depth':24,'num_heads':24,'init_values':1e-5,
                 'embed_dim':1536,'mlp_ratio':2.66667*2,'num_classes':0,'no_embed_class':True,
                 'mlp_layer':_MLP,'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True}
        self.uni_encoder=timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_cfg)

        cfg=AutoConfig.from_pretrained(dino_model)
        self.dino_model=AutoModel.from_pretrained(
            dino_model,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            low_cpu_mem_usage=True
        )

        self.proj_uni=nn.Sequential(nn.LayerNorm(self.uni_encoder.embed_dim), nn.Linear(self.uni_encoder.embed_dim, 512))
        self.proj_ctx=nn.Sequential(nn.LayerNorm(int(cfg.hidden_size)), nn.Linear(int(cfg.hidden_size), 512))
        self.head=nn.Sequential(nn.Dropout(0.2), nn.Linear(1024,1024), nn.SiLU(), nn.Dropout(0.2), nn.Linear(1024, num_classes))

    def load_from_ckpt(self, ckpt_path: str):
        sd_raw = torch.load(ckpt_path, map_location="cpu")
        state = sd_raw.get("state_dict", sd_raw)
        keep={}
        for k,v in state.items():
            if k.startswith(("uni_encoder.","proj_uni.","proj_ctx.","dino_model.","head_B.")):
                keep[k.replace("head_B.","head.")]=v
        self.load_state_dict(keep, strict=False)

    def _ctx_vec(self, xctx):
        B=xctx.shape[0]; mb=max(1, min(self.ctx_microbatch, B))
        outs=[]
        for i in range(0,B,mb):
            xi=xctx[i:i+mb].to(dtype=self.dino_model.dtype, non_blocking=True)
            out=self.dino_model(pixel_values=xi, return_dict=True)
            if self.use_pooler and hasattr(out,"pooler_output") and out.pooler_output is not None:
                vec=out.pooler_output
            else:
                vec=out.last_hidden_state[:,1:,:].mean(1)
            outs.append(vec)
        return torch.cat(outs, dim=0)

    def forward(self, x224, xctx):
        u=self.uni_encoder(x224)            # [B,1536]
        v=self._ctx_vec(xctx)               # [B,hidden]
        u_proj=self.proj_uni(u)             # [B,512]
        v_proj=self.proj_ctx(v)             # [B,512]
        fused=torch.cat([u_proj, v_proj], dim=1)  # [B,1024]
        logits=self.head(fused)             # [B,C]
        return logits, fused

# ==================== Gate LightningModule ====================
class FuseABGate(pl.LightningModule):
    def __init__(self, config, num_classes, num_tissues, class_names, ckptA, ckptB):
        super().__init__()
        self.save_hyperparameters(ignore=['class_names'])
        self.class_names=class_names
        self.num_classes=num_classes

        mean=torch.tensor([0.485,0.456,0.406], dtype=torch.float32).view(1,3,1,1)
        std=torch.tensor([0.229,0.224,0.225], dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

        # A/B
        self.modelA=PathA_InferExact(num_classes, num_tissues, tdim=self.hparams.config["tissue_embedding_dim"])
        self.modelA.load_from_ckpt(ckptA)
        self.modelB=PathB_InferExact(num_classes, dino_model=self.hparams.config["dino_model"],
                                     use_pooler=self.hparams.config["use_pooler"], ctx_microbatch=self.hparams.config.get("ctx_microbatch",16))
        self.modelB.load_from_ckpt(ckptB)

        # optional: TF32 + Flash SDP
        try:
            if self.hparams.config.get("enable_tf32", True) and torch.cuda.is_available():
                torch.set_float32_matmul_precision("high")
                torch.backends.cuda.matmul.allow_tf32 = True
                from torch.backends.cuda import sdp_kernel
                sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        except Exception:
            pass

        # start frozen; later unfreeze according to schedule
        for p in self.modelA.parameters(): p.requires_grad=False
        for p in self.modelB.parameters(): p.requires_grad=False
        self.modelA.eval(); self.modelB.eval()

        # gate: projections + simple statistics
        gdimA=int(self.hparams.config["gate_proj_dim_A"])
        gdimB=int(self.hparams.config["gate_proj_dim_B"])
        self.gate_proj_A = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, gdimA), nn.SiLU(), nn.Dropout(self.hparams.config["gate_drop"]))
        self.gate_proj_B = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, gdimB), nn.SiLU(), nn.Dropout(self.hparams.config["gate_drop"]))
        stat_dim = 8  # pAmax,pBmax,entA,entB,marA,marB,delta,|delta|
        in_dim = gdimA + gdimB + stat_dim
        h1, h2 = self.hparams.config["gate_hidden"]
        raw_gate_mlp = nn.Sequential(nn.Linear(in_dim, h1), nn.SiLU(), nn.Dropout(self.hparams.config["gate_drop"]),
                                     nn.Linear(h1, h2), nn.SiLU(), nn.Dropout(self.hparams.config["gate_drop"]),
                                     nn.Linear(h2, 1))

        # compile gate/heads if enabled
        if self.hparams.config.get("compile_gate", True):
            self.gate_mlp = _maybe_compile(raw_gate_mlp, "gate_mlp", self.hparams.config)
        else:
            self.gate_mlp = raw_gate_mlp
        if self.hparams.config.get("compile_heads", True):
            self.modelA.head = _maybe_compile(self.modelA.head, "head_A", self.hparams.config)
            self.modelB.head = _maybe_compile(self.modelB.head, "head_B", self.hparams.config)

        # optionally compile encoders (default False)
        if self.hparams.config.get("compile_encoders", False):
            self.modelA.uni_encoder = _maybe_compile(self.modelA.uni_encoder, "UNI2h_A", self.hparams.config)
            self.modelB.uni_encoder = _maybe_compile(self.modelB.uni_encoder, "UNI2h_B", self.hparams.config)
            self.modelB.dino_model  = _maybe_compile(self.modelB.dino_model,  "DINO_ctx", self.hparams.config)

        self.val_buf = dict(a_probs=[], b_probs=[], mix_probs=[], labels=[], gates=[])
        self._gate_only_epochs = float(self.hparams.config.get("gate_warmup_epochs", 0.0))
        self.safety_policy = None

    # --------- helpers ----------
    def _norm_inplace(self, t):
        if t.dtype!=torch.float32: t=t.to(dtype=torch.float32, non_blocking=True)
        t.mul_(1.0/255.0)
        mean=self.pixel_mean.to(t.device, non_blocking=True); std=self.pixel_std.to(t.device, non_blocking=True)
        t.sub_(mean).div_(std)
        return t

    @staticmethod
    def _stats_from_probs(p):
        pmax = p.max(dim=1).values
        top2,_ = p.topk(k=min(2, p.shape[1]), dim=1)
        margin = (top2[:,0] - (top2[:,1] if top2.shape[1]>1 else 0.0))
        ent = -(p.clamp_min(1e-8).log()*p).sum(dim=1) / math.log(p.shape[1])
        return pmax, ent, margin

    def _ab_trainable(self)->bool:
        if any(p.requires_grad for p in self.modelA.head.parameters()): return True
        if any(p.requires_grad for p in self.modelB.head.parameters()): return True
        if any(p.requires_grad for p in self.modelB.proj_uni.parameters()): return True
        if any(p.requires_grad for p in self.modelB.proj_ctx.parameters()): return True
        if hasattr(self.modelA.uni_encoder, "blocks"):
            if any(any(p.requires_grad for p in b.parameters()) for b in self.modelA.uni_encoder.blocks): return True
        if hasattr(self.modelB.uni_encoder, "blocks"):
            if any(any(p.requires_grad for p in b.parameters()) for b in self.modelB.uni_encoder.blocks): return True
        if hasattr(self.modelB, "dino_model") and hasattr(self.modelB.dino_model, "encoder") and hasattr(self.modelB.dino_model.encoder, "layer"):
            if any(any(p.requires_grad for p in b.parameters()) for b in self.modelB.dino_model.encoder.layer): return True
        return False

    def _forward_AB(self, x224, xctx, tissue):
        x224=self._norm_inplace(x224).contiguous(memory_format=torch.channels_last)
        xctx=self._norm_inplace(xctx).contiguous(memory_format=torch.channels_last)
        cg = nullcontext() if self._ab_trainable() else torch.no_grad()
        with cg:
            zA, featA = self.modelA(x224, tissue, return_feat=True)
            zB, fusedB = self.modelB(x224, xctx)
        pA = torch.softmax(zA.float(), dim=1)
        pB = torch.softmax(zB.float(), dim=1)
        return zA, zB, pA, pB, featA, fusedB

    def _maybe_unfreeze(self):
        if not self.hparams.config.get("joint_train", True): return
        if self.current_epoch < self._gate_only_epochs: return
        u = self.hparams.config.get("unfreeze", {})

        def set_train(m):
            m.train()
            for p in m.parameters(): p.requires_grad=True

        if u.get("A_head", False):
            set_train(self.modelA.head); u["A_head"]=False
        if u.get("B_head", False):
            set_train(self.modelB.head); u["B_head"]=False
        if u.get("B_proj", False):
            set_train(self.modelB.proj_uni); set_train(self.modelB.proj_ctx); u["B_proj"]=False

        k = int(u.get("A_last_k_blocks", 0))
        if k>0 and hasattr(self.modelA.uni_encoder, "blocks"):
            for blk in list(self.modelA.uni_encoder.blocks)[-k:]: set_train(blk)
            u["A_last_k_blocks"]=0

        k_uni = int(u.get("B_uni_last_k_blocks", 0))
        if k_uni>0 and hasattr(self.modelB.uni_encoder, "blocks"):
            for blk in list(self.modelB.uni_encoder.blocks)[-k_uni:]: set_train(blk)
            u["B_uni_last_k_blocks"]=0

        k_dino = int(u.get("B_dino_last_k_blocks", 0))
        if k_dino>0 and hasattr(self.modelB.dino_model, "encoder") and hasattr(self.modelB.dino_model.encoder, "layer"):
            for blk in list(self.modelB.dino_model.encoder.layer)[-k_dino:]: set_train(blk)
            u["B_dino_last_k_blocks"]=0

    # --------- forward / fusion ----------
    def forward(self, x224, xctx, tissue):
        zA, zB, pA, pB, featA, fusedB = self._forward_AB(x224, xctx, tissue)
        pAmax, entA, marA = self._stats_from_probs(pA)
        pBmax, entB, marB = self._stats_from_probs(pB)
        fA = self.gate_proj_A(featA)
        fB = self.gate_proj_B(fusedB)
        delta = (pBmax - pAmax)
        gate_in = torch.cat([
            fA, fB,
            pAmax[:,None], pBmax[:,None],
            entA[:,None], entB[:,None],
            marA[:,None], marB[:,None],
            delta[:,None], delta.abs()[:,None],
        ], dim=1)
        g_logit = self.gate_mlp(gate_in).squeeze(1)
        g = torch.sigmoid(g_logit)
        p_mix = (1.0 - g)[:,None]*pA + g[:,None]*pB
        return (zA, zB, pA, pB, p_mix, g_logit, g)

    # --------- Lightning hooks ----------
    def training_step(self, batch, batch_idx):
        x224,xctx,y,tissue = batch
        if self.hparams.config.get("gpu_resize_ctx", False):
            if xctx.shape[-1]!=self.hparams.config["ctx_img_size"] or xctx.shape[-2]!=self.hparams.config["ctx_img_size"]:
                xctx = F.interpolate(xctx.float(), size=(self.hparams.config["ctx_img_size"],)*2,
                                     mode="bilinear", align_corners=False).to(x224.dtype)

        self._maybe_unfreeze()
        zA, zB, pA, pB, pM, g_logit, g = self(x224,xctx,tissue)

        predA = pA.argmax(1); predB = pB.argmax(1)
        pyA = pA[torch.arange(y.shape[0], device=y.device), y]
        pyB = pB[torch.arange(y.shape[0], device=y.device), y]
        chooseB = ((predB==y) & (predA!=y))
        chooseA = ((predA==y) & (predB!=y))
        soft = torch.sigmoid(self.hparams.config["soft_k"]*(pyB - pyA))
        gate_t = torch.where(chooseB, torch.ones_like(soft),
                  torch.where(chooseA, torch.zeros_like(soft), soft)).detach()

        conflict = (chooseA ^ chooseB).float()
        w = 1.0 + self.hparams.config["gate_conflict_boost"] * conflict
        delta = (pyB - pyA).abs()
        w = w + self.hparams.config["gate_delta_scale"] * (delta ** self.hparams.config["gate_delta_gamma"])

        eps = 1e-8
        loss_mix = F.nll_loss((pM + eps).log(), y, reduction='none')
        loss_mix = (loss_mix * w).mean() * self.hparams.config["lambda_mix_ce"]

        if isinstance(self.hparams.config.get("gate_pos_weight","auto"), str):
            p_pos = gate_t.mean().clamp(1e-3, 1-1e-3).detach()
            pos_w = (1.0 - p_pos) / p_pos
        else:
            pos_w = torch.tensor(float(self.hparams.config["gate_pos_weight"]), device=g_logit.device)
        loss_bce = F.binary_cross_entropy_with_logits(g_logit, gate_t, weight=w, pos_weight=pos_w)
        loss_bce = loss_bce * self.hparams.config["lambda_gate_bce"]

        p_sel = torch.where((gate_t>0.5)[:,None], pB, pA)
        kl = (p_sel * (p_sel.add(eps).log() - pM.add(eps).log())).sum(dim=1)
        loss_align = (kl * w).mean() * self.hparams.config["gate_lambda_align"]

        g_clamp = g.clamp(1e-6, 1-1e-6)
        ent_g = -(g_clamp.log()*g_clamp + (1-g_clamp).log()*(1-g_clamp))
        loss_ent = (ent_g * delta.detach()).mean() * self.hparams.config["gate_lambda_entropy"]

        loss_aux = 0.0
        if self._ab_trainable() and self.hparams.config.get("lambda_aux_heads", 0.0) > 0:
            loss_aux = (F.cross_entropy(zA, y) + F.cross_entropy(zB, y)) * (self.hparams.config["lambda_aux_heads"] * 0.5)

        loss = loss_mix + loss_bce + loss_align + loss_ent + loss_aux

        acc_mix = (pM.argmax(1)==y).float().mean()
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/loss_mix", loss_mix, on_step=True)
        self.log("train/loss_gate_bce", loss_bce, on_step=True)
        self.log("train/loss_align", loss_align, on_step=True)
        self.log("train/loss_entropy", loss_ent, on_step=True)
        self.log("train/loss_aux", loss_aux, on_step=True)
        self.log("train/acc_mix", acc_mix, on_step=True, prog_bar=True)
        self.log("gate/mean_g", g.mean(), on_step=True)
        self.log("gate/target_mean", gate_t.mean(), on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x224,xctx,y,tissue = batch
        if self.hparams.config.get("gpu_resize_ctx", False):
            if xctx.shape[-1]!=self.hparams.config["ctx_img_size"] or xctx.shape[-2]!=self.hparams.config["ctx_img_size"]:
                xctx = F.interpolate(xctx.float(), size=(self.hparams.config["ctx_img_size"],)*2,
                                     mode="bilinear", align_corners=False).to(x224.dtype)

        with torch.no_grad():
            _, _, pA, pB, pM, _, g = self(x224,xctx,tissue)

        self.val_buf["a_probs"].append(pA.detach().float().cpu())
        self.val_buf["b_probs"].append(pB.detach().float().cpu())
        self.val_buf["mix_probs"].append(pM.detach().float().cpu())
        self.val_buf["labels"].append(y.detach().long().cpu())
        self.val_buf["gates"].append(g.detach().float().cpu())
        logits_like = (pM.clamp_min(1e-8)).log()
        return F.nll_loss(logits_like, y)

    def _metrics_block(self, probs_t: torch.Tensor, labels_t: torch.Tensor):
        probs = probs_t.float()
        preds = probs.argmax(1)
        y_true = labels_t.long().numpy()
        y_pred = preds.numpy()
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        per_f1 = f1_score(y_true, y_pred, average=None, labels=np.arange(self.num_classes), zero_division=0)
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        return acc, f1m, per_f1, cm, probs, preds

    # >>> SAFETY: reliability and threshold calibration
    @staticmethod
    def _reliability_by_pred_class(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        pred = probs.argmax(1); C = probs.shape[1]
        rel = np.zeros(C, dtype=np.float32)
        for c in range(C):
            idx = np.where(pred==c)[0]
            if idx.size>0:
                rel[c] = (y[idx]==c).mean()
        return rel

    @staticmethod
    def _baseline_choose(a_probs: np.ndarray, b_probs: np.ndarray, relA: np.ndarray, relB: np.ndarray) -> np.ndarray:
        a_max = a_probs.max(1); a_cls = a_probs.argmax(1)
        b_max = b_probs.max(1); b_cls = b_probs.argmax(1)
        scoreA = a_max * (relA[a_cls] + 1e-6)
        scoreB = b_max * (relB[b_cls] + 1e-6)
        chooseB = scoreB > scoreA
        return np.where(chooseB[:,None], b_probs, a_probs)

    @staticmethod
    def _apply_safe_policy(a_probs: np.ndarray, b_probs: np.ndarray, g: np.ndarray,
                           relA: np.ndarray, relB: np.ndarray, tau: float, gamma: float) -> np.ndarray:
        a_max = a_probs.max(1); b_max = b_probs.max(1)
        r = b_max - a_max
        base = FuseABGate._baseline_choose(a_probs, b_probs, relA, relB)
        chooseB = (g > tau) & (r > gamma)
        chooseA = (g < (1.0 - tau)) & (r < -gamma)
        out = base.copy()
        out[chooseB] = b_probs[chooseB]
        out[chooseA] = a_probs[chooseA]
        return out

    @staticmethod
    def _to_numpy(x, kind="float"):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if kind == "float":
                return x.float().numpy()
            elif kind == "long":
                return x.long().numpy()
            else:
                return x.numpy()
        x = np.asarray(x)
        if kind == "float":
            return x.astype(np.float32, copy=False)
        elif kind == "long":
            return x.astype(np.int64, copy=False)
        else:
            return x

    def _calibrate_safety(self, a_probs, b_probs, g, y):
        a_np = self._to_numpy(a_probs, "float")
        b_np = self._to_numpy(b_probs, "float")
        g_np = self._to_numpy(g, "float").reshape(-1)
        y_np = self._to_numpy(y, "long").reshape(-1)

        accA = float(accuracy_score(y_np, a_np.argmax(1)))
        accB = float(accuracy_score(y_np, b_np.argmax(1)))
        best_single = max(accA, accB)

        relA = self._reliability_by_pred_class(a_np, y_np)
        relB = self._reliability_by_pred_class(b_np, y_np)
        base_probs = self._baseline_choose(a_np, b_np, relA, relB)
        acc_base = float(accuracy_score(y_np, base_probs.argmax(1)))

        taus = np.linspace(0.3, 0.7, 9)
        gammas = np.linspace(0.0, 0.20, 11)
        best = {"acc_safe": acc_base, "tau": None, "gamma": None, "mode": "baseline",
                "acc_base": acc_base, "best_single": best_single}

        for tau in taus:
            for gamma in gammas:
                probs = self._apply_safe_policy(a_np, b_np, g_np, relA, relB, float(tau), float(gamma))
                acc = float(accuracy_score(y_np, probs.argmax(1)))
                if acc >= best_single and acc > best["acc_safe"]:
                    best.update({"acc_safe": acc, "tau": float(tau), "gamma": float(gamma), "mode":"safe"})

        return {"relA": relA.tolist(), "relB": relB.tolist(), **best}

    def on_validation_epoch_end(self):
        if not self.val_buf["mix_probs"]: return
        a_probs=torch.cat(self.val_buf["a_probs"])
        b_probs=torch.cat(self.val_buf["b_probs"])
        mix_probs=torch.cat(self.val_buf["mix_probs"])
        y=torch.cat(self.val_buf["labels"]).long()
        gates=torch.cat(self.val_buf["gates"]).float()

        accA,f1A,_,_,_,_ = self._metrics_block(a_probs, y)
        accB,f1B,_,_,_,_ = self._metrics_block(b_probs, y)
        accM,f1M,per_f1M,cmM,_,_ = self._metrics_block(mix_probs, y)

        pyA=a_probs[torch.arange(y.shape[0]), y]; pyB=b_probs[torch.arange(y.shape[0]), y]
        predA = a_probs.argmax(1); predB = b_probs.argmax(1)
        betterB = ((predB==y) & (predA!=y))
        betterA = ((predA==y) & (predB!=y))
        chooseB = torch.where(betterA, torch.zeros_like(betterA), torch.where(betterB, torch.ones_like(betterB), (pyB>pyA)))
        oracle_probs = torch.where(chooseB[:,None], b_probs, a_probs)
        accO,f1O,_,_,_,_ = self._metrics_block(oracle_probs, y)

        A_correct = (a_probs.argmax(1) == y)
        B_correct = (b_probs.argmax(1) == y)
        both   = A_correct & B_correct
        a_only = A_correct & (~B_correct)
        b_only = (~A_correct) & B_correct
        N = float(y.numel())
        oracle_acc_label = float((both.sum() + a_only.sum() + b_only.sum()) / N)
        complementarity_gain = oracle_acc_label - max(float(A_correct.float().mean()), float(B_correct.float().mean()))
        conflict_rate = float((a_only.sum() + b_only.sum()) / N)

        chooseB_gate = (gates > 0.5)
        true_B_better = b_only
        true_A_better = a_only
        choose_B_precision = float(((chooseB_gate & true_B_better).sum() / max(1, chooseB_gate.sum())).cpu().item())
        choose_A_precision = float((((~chooseB_gate) & true_A_better).sum() / max(1, (~chooseB_gate).sum())).cpu().item())
        mix_minus_best = accM - max(accA, accB)

        # SAFETY
        safe = self._calibrate_safety(a_probs.float(), b_probs.float(), gates.float(), y)
        self.safety_policy = safe
        acc_safe = safe["acc_safe"]; acc_base = safe["acc_base"]; best_single = safe["best_single"]
        safe_minus_best = acc_safe - best_single

        # logging
        self.log("val/acc_A", accA, prog_bar=True)
        self.log("val/acc_B", accB, prog_bar=True)
        self.log("val/acc_mix", accM, prog_bar=True)
        self.log("val/acc_oracle", accO, prog_bar=True)
        self.log("val/f1_macro_mix", f1M, prog_bar=True)
        self.log("val/oracle_acc_label", oracle_acc_label, prog_bar=True)
        self.log("val/complementarity_gain", complementarity_gain, prog_bar=True)
        self.log("val/conflict_rate", conflict_rate)
        self.log("gate/mean_g", gates.mean())
        self.log("gate/choose_B_precision", choose_B_precision)
        self.log("gate/choose_A_precision", choose_A_precision)
        self.log("val/mix_minus_best", mix_minus_best, prog_bar=True)
        self.log("val/acc_safe", acc_safe, prog_bar=True)
        self.log("val/acc_safe_minus_best", safe_minus_best, prog_bar=True)
        self.log("val/acc_safe_baseline", acc_base)

        # save safety policy JSON
        try:
            outdir = self.trainer.default_root_dir or self.hparams.config.get("checkpoint_dir")
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "safety_policy.json"), "w") as f:
                json.dump(safe, f, indent=2)
        except Exception as e:
            print(f"[WARN] save safety_policy.json failed: {e}")

        if isinstance(self.logger, pl.loggers.WandbLogger) and self.trainer.is_global_zero:
            import wandb
            fig_cm, ax = plt.subplots(figsize=(12,10))
            sns.heatmap(cmM, annot=True, fmt=".2%", cmap="Blues",
                        xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
            ax.set_title(f"Fusion Mix Confusion (epoch {self.current_epoch})"); ax.tick_params(axis='x', rotation=45); plt.tight_layout()
            self.logger.log_image(key="val/confusion_mix", images=[fig_cm]); plt.close(fig_cm)

            fig_bar, ax2 = plt.subplots(figsize=(6,4))
            xs=["A","B","Mix","Oracle","Safe"]; ys=[accA,accB,accM,accO,acc_safe]
            ax2.bar(xs, ys); ax2.set_ylim(0,1); ax2.set_ylabel("Accuracy"); ax2.set_title("Acc & Oracle & Safe")
            for i,v in enumerate(ys): ax2.text(i, v+0.01, f"{v:.3f}", ha='center')
            plt.tight_layout(); self.logger.log_image(key="val/oracle_bar", images=[fig_bar]); plt.close(fig_bar)

        for k in self.val_buf: self.val_buf[k].clear()

    def configure_optimizers(self):
        params_gate = list(self.gate_proj_A.parameters()) + list(self.gate_proj_B.parameters()) + list(self.gate_mlp.parameters())
        pg = [{"params": params_gate, "lr": self.hparams.config["lr_gate"], "weight_decay": self.hparams.config["weight_decay"]}]

        if any(p.requires_grad for p in self.modelA.head.parameters()):
            pg.append({"params": self.modelA.head.parameters(), "lr": self.hparams.config["lr_heads"], "weight_decay": self.hparams.config["weight_decay"]})
        if any(p.requires_grad for p in self.modelB.head.parameters()):
            pg.append({"params": self.modelB.head.parameters(), "lr": self.hparams.config["lr_heads"], "weight_decay": self.hparams.config["weight_decay"]})
        if any(p.requires_grad for p in self.modelB.proj_uni.parameters()):
            pg.append({"params": self.modelB.proj_uni.parameters(), "lr": self.hparams.config["lr_heads"], "weight_decay": self.hparams.config["weight_decay"]})
        if any(p.requires_grad for p in self.modelB.proj_ctx.parameters()):
            pg.append({"params": self.modelB.proj_ctx.parameters(), "lr": self.hparams.config["lr_heads"], "weight_decay": self.hparams.config["weight_decay"]})

        def add_if_trainable(m):
            ps=[p for p in m.parameters() if p.requires_grad]
            if ps: pg.append({"params": ps, "lr": self.hparams.config["lr_backbone"], "weight_decay": self.hparams.config["weight_decay"]})
        if hasattr(self.modelA.uni_encoder, "blocks"):
            add_if_trainable(nn.ModuleList([b for b in self.modelA.uni_encoder.blocks if any(p.requires_grad for p in b.parameters())]))
        if hasattr(self.modelB.uni_encoder, "blocks"):
            add_if_trainable(nn.ModuleList([b for b in self.modelB.uni_encoder.blocks if any(p.requires_grad for p in b.parameters())]))
        if hasattr(self.modelB, "dino_model") and hasattr(self.modelB.dino_model, "encoder") and hasattr(self.modelB.dino_model.encoder, "layer"):
            add_if_trainable(nn.ModuleList([b for b in self.modelB.dino_model.encoder.layer if any(p.requires_grad for p in b.parameters())]))

        try:
            opt = torch.optim.AdamW(pg, fused=bool(self.hparams.config.get("optimizer_fused", True)))
        except TypeError:
            opt = torch.optim.AdamW(pg)

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: min(1.0, (step+1)/max(1, self.hparams.config["warmup_steps"])))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval":"step"}}

# ==================== Inference dump (with safety gate) ====================
@torch.no_grad()
def run_inference_and_dump_fusion(model_ckpt_path, config, dm: PathBDataModule, class_names: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FuseABGate.load_from_checkpoint(
        checkpoint_path=model_ckpt_path,
        config=config, num_classes=len(class_names), num_tissues=dm.num_tissues, class_names=class_names,
        ckptA=config["ckpt_pathA"], ckptB=config["ckpt_pathB"],
        map_location=device
    ).to(device).eval()

    loader = dm.val_dataloader()
    a_probs=[]; b_probs=[]; mix_probs=[]; labels=[]; gates=[]
    for batch in loader:
        x224,xctx,y,tissue = batch if isinstance(batch, tuple) else (batch[0],batch[1],batch[2],batch[3])
        x224=x224.to(device, non_blocking=True); xctx=xctx.to(device, non_blocking=True); y=y.to(device, non_blocking=True); tissue=tissue.to(device, non_blocking=True)
        if config.get("gpu_resize_ctx", False):
            if xctx.shape[-1]!=config["ctx_img_size"] or xctx.shape[-2]!=config["ctx_img_size"]:
                xctx = F.interpolate(xctx.float(), size=(config["ctx_img_size"],)*2,
                                     mode="bilinear", align_corners=False).to(x224.dtype)
        _,_,pA,pB,pM,_,g = model(x224,xctx,tissue)
        a_probs.append(pA.cpu()); b_probs.append(pB.cpu()); mix_probs.append(pM.cpu())
        labels.append(y.cpu()); gates.append(g.cpu())

    a_probs=torch.cat(a_probs).float().numpy()
    b_probs=torch.cat(b_probs).float().numpy()
    mix_probs=torch.cat(mix_probs).float().numpy()
    labels_np=np.array(torch.cat(labels).long().numpy())
    gates_np=np.array(torch.cat(gates).float().numpy())

    def _eval(probs, labels_np):
        preds = probs.argmax(1)
        acc = float(accuracy_score(labels_np, preds))
        f1m = float(f1_score(labels_np, preds, average='macro', zero_division=0))
        return preds, acc, f1m

    predA, accA, f1A = _eval(a_probs, labels_np)
    predB, accB, f1B = _eval(b_probs, labels_np)
    predM, accM, f1M = _eval(mix_probs, labels_np)

    pyA=a_probs[np.arange(len(labels_np)), labels_np]; pyB=b_probs[np.arange(len(labels_np)), labels_np]
    chooseB = (predB==labels_np) & ((predA!=labels_np) | ((predA==labels_np)&(pyB>pyA)))
    pO = np.where(chooseB[:,None], b_probs, a_probs)
    predO = np.argmax(pO,1)
    accO = float(accuracy_score(labels_np, predO)); f1O = float(f1_score(labels_np, predO, average='macro', zero_division=0))

    policy_path = os.path.join(os.path.dirname(model_ckpt_path), "safety_policy.json")
    if os.path.exists(policy_path):
        safe = json.load(open(policy_path, "r"))
    else:
        relA = FuseABGate._reliability_by_pred_class(a_probs, labels_np).tolist()
        relB = FuseABGate._reliability_by_pred_class(b_probs, labels_np).tolist()
        tmp = FuseABGate._apply_safe_policy(a_probs, b_probs, gates_np, np.array(relA), np.array(relB), 0.5, 0.05)
        acc_base = float(accuracy_score(labels_np, tmp.argmax(1)))
        safe = {"relA": relA, "relB": relB, "tau": 0.5, "gamma": 0.05, "mode":"baseline",
                "acc_safe": acc_base, "best_single": max(accA,accB), "acc_base": acc_base}
    relA = np.array(safe["relA"]); relB = np.array(safe["relB"])
    if safe.get("mode","baseline")=="safe" and safe.get("tau") is not None:
        probs_safe = FuseABGate._apply_safe_policy(a_probs, b_probs, gates_np, relA, relB, safe["tau"], safe["gamma"])
    else:
        probs_safe = FuseABGate._baseline_choose(a_probs, b_probs, relA, relB)

    predS, accS, f1S = _eval(probs_safe, labels_np)

    A_correct = (predA == labels_np); B_correct = (predB == labels_np)
    both   = A_correct & B_correct
    a_only = A_correct & (~B_correct)
    b_only = (~A_correct) & B_correct
    N=float(len(labels_np))
    oracle_acc_label = float((both.sum() + a_only.sum() + b_only.sum()) / N)
    complementarity_gain = oracle_acc_label - max(float(A_correct.mean()), float(B_correct.mean()))
    conflict_rate = float((a_only.sum() + b_only.sum()) / N)

    np.save(os.path.join(out_dir,"probs_A.npy"), a_probs)
    np.save(os.path.join(out_dir,"probs_B.npy"), b_probs)
    np.save(os.path.join(out_dir,"probs_mix.npy"), mix_probs)
    np.save(os.path.join(out_dir,"probs_oracle.npy"), pO)
    np.save(os.path.join(out_dir,"probs_safe.npy"), probs_safe)
    np.save(os.path.join(out_dir,"preds_A.npy"), predA)
    np.save(os.path.join(out_dir,"preds_B.npy"), predB)
    np.save(os.path.join(out_dir,"preds_mix.npy"), predM)
    np.save(os.path.join(out_dir,"preds_oracle.npy"), predO)
    np.save(os.path.join(out_dir,"preds_safe.npy"), predS)
    np.save(os.path.join(out_dir,"labels.npy"), labels_np)
    np.save(os.path.join(out_dir,"gates.npy"), gates_np)

    with open(os.path.join(out_dir,"metrics.json"),"w") as f:
        json.dump({
            "acc_A":accA,"f1_A":f1A,"acc_B":accB,"f1_B":f1B,
            "acc_mix":accM,"f1_mix":f1M,
            "acc_oracle":accO,"f1_oracle":f1O,
            "acc_safe":accS,"f1_safe":f1S,
            "oracle_acc_label":oracle_acc_label,"complementarity_gain":complementarity_gain,
            "conflict_rate":conflict_rate,
            "safe_minus_best": accS - max(accA,accB)
        }, f, indent=2)

    print(f"[FUSE] A(acc={accA:.4f},f1={f1A:.4f}) | B(acc={accB:.4f},f1={f1B:.4f}) | Mix(acc={accM:.4f},f1={f1M:.4f})")
    print(f"[FUSE] Oracle(acc={accO:.4f},f1={f1O:.4f}) | Safe(acc={accS:.4f},f1={f1S:.4f}) | Safe- Best = {accS - max(accA,accB):+.4f}")
    print(f"[FUSE] complementarity: oracle_acc_label={oracle_acc_label:.4f} gain={complementarity_gain:.4f} conflict={conflict_rate:.4f}")
    print(f"[FUSE] dumps -> {out_dir}")

# ==================== Entry ====================
def main():
    config = prepare_config()

    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        accelerator, devices, precision = 'gpu', 1, 'bf16-mixed'
        print(f"[INFO] Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        print(f"[INFO] Training on cuda:0 -> {torch.cuda.get_device_name(0)}", flush=True)
    else:
        accelerator, devices, precision = 'cpu', 1, '32'
        print("[INFO] CUDA not available, using CPU.", flush=True)

    dm = PathBDataModule(config, NEW_LABEL_MAPPING, CLASSES_TO_REMOVE)
    dm.setup()

    model = FuseABGate(config=config, num_classes=dm.num_classes, num_tissues=dm.num_tissues,
                       class_names=dm.class_names, ckptA=config["ckpt_pathA"], ckptB=config["ckpt_pathB"])

    wandb_logger=None
    try:
        import pytorch_lightning.loggers as pl_loggers
        wandb_logger = pl_loggers.WandbLogger(
            name=config["experiment_name"], project=config["project_name"],
            entity=config["entity"], config=config, resume='allow'
        )
    except Exception as e:
        print(f"[WARN] WandB logger not available: {e}")

    ckpt_cb = ModelCheckpoint(
        monitor="val/acc_safe",
        mode="max",
        dirpath=config["checkpoint_dir"],
        filename="best-epoch={epoch}-val_accsafe={val/acc_safe:.4f}",
    )
    early_cb = EarlyStopping(monitor="val/acc_safe", patience=config["early_stopping_patience"], mode="max", verbose=True)

    trainer = Trainer(
        accelerator=accelerator, devices=devices, precision=precision, strategy='auto',
        max_epochs=config["max_epochs"],
        val_check_interval=config["val_check_interval"],
        gradient_clip_val=0.0,
        callbacks=[TQDMProgressBar(refresh_rate=20), early_cb, ckpt_cb],
        logger=wandb_logger, log_every_n_steps=20,
        accumulate_grad_batches=config["accumulate_grad_batches"], enable_checkpointing=True,
        limit_val_batches=config.get("limit_val_batches", None),
    )

    trainer.fit(model, datamodule=dm)
    best = ckpt_cb.best_model_path
    if best and os.path.exists(best):
        print(f"[FUSE] Best checkpoint: {best}")
        run_inference_and_dump_fusion(best, config, dm, dm.class_names, config["inference_out_dir"])
    else:
        print("[FUSE] No best checkpoint found; skip inference dump.")

if __name__ == "__main__":
    main()
