# -*- coding: utf-8 -*-
"""
Grad-CAM for MUSKLinearProbeFT (ViT/BEiT3 backbone + organ embedding + linear head).
Provide the annotations JSON, H5 path, and checkpoint to reproduce figures.
"""

import os, json, warnings, re, argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as T

from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- Register MUSK with timm -----
try:
    import musk, musk.modeling  # noqa: F401
except Exception as e:
    raise RuntimeError("Unable to import `musk`. Install it or add the source to PYTHONPATH.") from e

# ----- Config -----
DEFAULT_OUT_DIR = Path("results") / "figure_reproduce" / "grad_cam" / "musk"
DEFAULT_KEYS      = ["gidojinm-1","ijjnlofi-1","lldfeija-1","ignfhjao-1","oiimjodd-1"]

TISSUE_TO_IDX = {"breast":0,"colon":1,"heart":2,"kidney":3,"liver":4,"lung":5,"ovary":6,"pancreas":7}

MODEL_IMG_SIZE = 384   # Training ran at 224 -> upsampled to 384

# ----- H5 I/O -----
class H5Reader:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.f = None
        self.merged = False
        self.k2i = None

    def _ensure_open(self):
        if self.f is None:
            self.f = h5py.File(self.path, "r")
            if "images" in self.f and "keys" in self.f:
                try:
                    keys = self.f["keys"].asstr()[()]
                except Exception:
                    raw = self.f["keys"][()]
                    keys = [k.decode() if isinstance(k,(bytes,np.bytes_)) else str(k) for k in raw]
                self.k2i = {str(k): i for i, k in enumerate(keys)}
                self.merged = True

    def get(self, key: str) -> np.ndarray:
        self._ensure_open()
        key = str(key)
        if self.merged:
            i = self.k2i.get(key)
            if i is None:
                raise KeyError(f"[{self.path}] missing key in merged H5: {key}")
            arr = self.f["images"][i][()]
        else:
            if key not in self.f:
                raise KeyError(f"[{self.path}] missing key: {key}")
            arr = self.f[key][()]
        # Force uint8 224x224x3
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[...,:3]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape != (224,224,3):
            raise ValueError(f"unexpected shape for {key}: {arr.shape}")
        return arr

    def close(self):
        try:
            if self.f is not None:
                self.f.close()
        finally:
            self.f=None; self.merged=False; self.k2i=None

def load_annotations(json_path: str) -> Dict[str, str]:
    if not os.path.exists(json_path):
        return {}
    df = pd.read_json(json_path, orient="index").reset_index().rename(columns={"index":"cell_id"})
    label_col = None
    for k in ["class_id","label","cell_type","type","annotation"]:
        if k in df.columns:
            label_col = k
            break
    if label_col is None:
        return {}
    df["cell_id"] = df["cell_id"].astype(str)
    return dict(zip(df["cell_id"], df[label_col].astype(str)))

# ----- Helpers -----
def _to_hw_pair(v, default=16):
    if isinstance(v, (tuple, list)) and len(v) == 2: return int(v[0]), int(v[1])
    if isinstance(v, (int, float)): x = int(v); return x, x
    x = int(default); return x, x

def _try_fix_beit3_img_grid(backbone: nn.Module, target_img_size: int):
    """Align BEiT3/ViT image/grid settings with the target inference size."""
    try:
        beit3 = getattr(backbone, "beit3", None)
        ve = getattr(beit3, "vision_embed", None) if beit3 is not None else None
        if ve is not None:
            if hasattr(ve, "img_size"): ve.img_size = (target_img_size, target_img_size)
            ps_h, ps_w = _to_hw_pair(getattr(ve, "patch_size", 16), default=16)
            img_h, img_w = _to_hw_pair(getattr(ve, "img_size", (target_img_size, target_img_size)), default=target_img_size)
            gh, gw = img_h // ps_h, img_w // ps_w
            if hasattr(ve, "grid_size"): ve.grid_size = (gh, gw)
            if hasattr(ve, "num_patches"): ve.num_patches = gh * gw
        for cfg_attr in ("cfg","config"):
            cfg_obj = getattr(beit3, cfg_attr, None)
            if cfg_obj is None: continue
            for key in ("img_size","image_size"):
                if hasattr(cfg_obj, key): setattr(cfg_obj, key, (target_img_size, target_img_size))
    except Exception:
        pass

def _canon_key(k: str) -> str:
    """Strip common prefixes (module./model./net./pl_module.)."""
    for p in ("module.", "model.", "net.", "pl_module."):
        if k.startswith(p):
            return k[len(p):]
    return k

# ----- MUSK inference model -----
class MUSKInference(nn.Module):
    """Mirror training forward: backbone global feat -> concat organ -> head."""
    def __init__(self, backbone_name: str,
                 head_in_dim: int, head_out_dim: int,
                 num_organs: int, organ_dim: int = 32,
                 use_organ: bool = True, img_size: int = 384):
        super().__init__()
        self.backbone = create_model(backbone_name, pretrained=False)
        _try_fix_beit3_img_grid(self.backbone, img_size)
        self.use_organ = bool(use_organ)
        self.organ = nn.Embedding(num_organs, organ_dim) if self.use_organ else None
        self.head = nn.Linear(head_in_dim, head_out_dim)

    def forward(self, x: torch.Tensor, organs: torch.Tensor) -> torch.Tensor:
        # Align with MUSK backbone API (BEiT3)
        feats = self.backbone(image=x, with_head=False, out_norm=False, ms_aug=False, return_global=True)[0]
        if self.use_organ:
            feats = torch.cat([feats, self.organ(organs)], dim=1)
        return self.head(feats)

def build_model_from_musk_ckpt(ckpt_path: str, device: torch.device) -> Tuple[MUSKInference, List[str]]:
    obj = torch.load(ckpt_path, map_location="cpu")
    sd_raw = obj.get("state_dict", obj)
    hparams = obj.get("hyper_parameters", {})

    # Canonicalize keys
    sd = { _canon_key(k): v for k, v in sd_raw.items() }

    # Pick head.weight (prefer the top-level head over backbone.head)
    cand = [k for k in sd.keys() if k.endswith("head.weight")]
    if not cand:
        raise RuntimeError("No head.weight found in checkpoint.")
    # Prefer exact 'head.weight'; otherwise choose the shortest non-backbone key
    key_head = None
    if "head.weight" in sd:
        key_head = "head.weight"
    else:
        non_backbone = [k for k in cand if not k.startswith("backbone.")]
        cand2 = non_backbone if non_backbone else cand
        key_head = sorted(cand2, key=lambda s: (s.count("."), len(s)))[0]
    W = sd[key_head]
    head_out_dim = int(W.shape[0])   # num_classes
    head_in_dim  = int(W.shape[1])   # global feat (+ organ embed)
    print(f"[ckpt] choose head key: '{key_head}'  shape={tuple(W.shape)}  -> out={head_out_dim}, in={head_in_dim}")

    # Backbone name
    cfg = hparams.get("cfg", {}) or hparams.get("config", {}) or {}
    backbone_name = cfg.get("musk_backbone", "musk_large_patch16_384")

    # class names
    class_names = hparams.get("class_names", None)
    if (class_names is None) or (len(class_names) != head_out_dim):
        class_names = [f"cls_{i}" for i in range(head_out_dim)]

    # organ embedding
    organ_key = None
    for k in ("organ.weight", "organ_embed.weight", "organ_embedding.weight"):
        if k in sd and sd[k].ndim == 2:
            organ_key = k; break
    if organ_key is not None:
        num_organs = int(sd[organ_key].shape[0])
        organ_dim  = int(sd[organ_key].shape[1])
        use_organ  = True
        print(f"[ckpt] organ embed: key='{organ_key}'  shape={tuple(sd[organ_key].shape)}")
    else:
        num_organs = int(cfg.get("num_organs", 8))
        organ_dim  = int(cfg.get("organ_embedding_dim", 32))
        use_organ  = bool(cfg.get("use_organ_embed", True))
        print(f"[ckpt] organ embed: not found in ckpt, use cfg -> num_organs={num_organs}, dim={organ_dim}, use={use_organ}")

    # Build inference model with the checkpoint head shape
    model = MUSKInference(
        backbone_name=backbone_name,
        head_in_dim=head_in_dim, head_out_dim=head_out_dim,
        num_organs=num_organs, organ_dim=organ_dim, use_organ=use_organ,
        img_size=MODEL_IMG_SIZE,
    )

    # Load only backbone.* (excluding backbone.head.*), organ.*, head.*
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            if k.startswith("backbone.head."):  # Skip backbone's own head
                continue
            new_sd[k] = v
        elif k.startswith("head."):
            new_sd[k] = v
        elif k.startswith("organ.") or k.startswith("organ_embed.") or k.startswith("organ_embedding."):
            # Normalize to 'organ.'
            new_sd["organ." + k.split(".", 1)[1]] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")

    model.to(device).eval()
    for p in model.parameters():  # Grad-CAM needs gradients
        p.requires_grad = True
    return model, class_names

# ----- Grad-CAM (first Conv2d) -----
class GradCAMonFirstConv:
    """Target layer: first Conv2d (ViT/BEiT patch embedding).
    CAM = ReLU(sum_k(mean_{HW}(dY/dA_k) * A_k)) resampled 384->224."""
    def __init__(self, model: MUSKInference, device: torch.device):
        self.model = model
        self.device = device
        self.target_layer = self._find_first_conv(model.backbone)
        if self.target_layer is None:
            raise RuntimeError("Could not find a Conv2d layer in the MUSK backbone for Grad-CAM.")
        self._f = None
        self._handles = []

    @staticmethod
    def _find_first_conv(module: nn.Module):
        for name in ["beit3.vision_embed.proj", "patch_embed.proj"]:
            obj = module
            ok = True
            for attr in name.split("."):
                if not hasattr(obj, attr):
                    ok = False; break
                obj = getattr(obj, attr)
            if ok and isinstance(obj, nn.Conv2d):
                return obj
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m
        return None

    def _register(self):
        def fwd_hook(_m, _in, out):
            self._f = out
            try:
                self._f.retain_grad()
            except Exception:
                pass
        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))

    def _remove(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles.clear()

    @torch.no_grad()
    def _to_224(self, cam_384: torch.Tensor) -> np.ndarray:
        cam = torch.nn.functional.interpolate(cam_384, size=(224,224), mode="bilinear", align_corners=False)
        cam = cam[0,0]
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-6)
        return cam.detach().cpu().numpy().astype(np.float32)

    def __call__(self, x384: torch.Tensor, tissue_id: int, target_index: int = None):
        self._register()
        try:
            organs = torch.tensor([tissue_id], dtype=torch.long, device=self.device)
            logits = self.model(x384, organs)  # Forward pass; hook captures self._f (B,C,H',W')
            probs  = torch.softmax(logits, dim=1)
            pred_prob, pred_idx = float(probs[0].max().item()), int(probs[0].argmax().item())
            if target_index is None:
                target_index = pred_idx

            # Backprop
            self.model.zero_grad(set_to_none=True)
            if self._f is not None and self._f.grad is not None:
                self._f.grad.zero_()
            logits[0, target_index].backward(retain_graph=False)

            feat = self._f.detach()                        # [1, C, H', W']
            grad = self._f.grad
            if grad is None:
                raise RuntimeError("Missing gradients for the target layer (check requires_grad/eval state).")

            weights = torch.mean(grad, dim=(2,3), keepdim=True)   # [1, C, 1, 1]
            cam = torch.sum(weights * feat, dim=1, keepdim=True)  # [1, 1, H', W']
            cam = torch.relu(cam)

            # Normalize, upsample to 384, then downsample to 224
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-6)
            cam_384 = torch.nn.functional.interpolate(cam, size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE),
                                                      mode="bilinear", align_corners=False)
            cam_224 = self._to_224(cam_384)
            return cam_224, pred_idx, pred_prob
        finally:
            self._remove()

# ----- Visualization -----
def save_overlay(img_u8_224: np.ndarray, cam_224: np.ndarray, out_png: str, alpha: float = 0.45):
    plt.figure(figsize=(3,3), dpi=224/3)
    plt.axis("off")
    plt.imshow(img_u8_224)
    plt.imshow(cam_224, cmap="jet", alpha=alpha)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

# ----- Main -----
def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM for MUSK linear probe.")
    parser.add_argument("--anno-json", required=True, help="Path to annotations.json")
    parser.add_argument("--h5-path", required=True, help="Path to patches_224x224.h5")
    parser.add_argument("--ckpt-path", required=True, help="Path to MUSK checkpoint (.ckpt)")
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, help="Cell keys to visualize")
    parser.add_argument("--tissue", default="ovary", help="Tissue name (default: ovary)")
    parser.add_argument("--tissue-id", type=int, default=None, help="Override tissue id explicitly")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Directory for Grad-CAM PNG outputs")
    parser.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_index)
        print("[CUDA]", torch.cuda.get_device_name(torch.cuda.current_device()))

    ann = load_annotations(args.anno_json)
    model, class_names = build_model_from_musk_ckpt(args.ckpt_path, device=device)
    print(f"[ckpt] backbone ready, classes={len(class_names)}")

    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    resize_224_to_384 = T.Resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC)

    reader = H5Reader(args.h5_path)
    cammer = GradCAMonFirstConv(model, device=device)

    tissue_name = args.tissue.lower()
    if args.tissue_id is not None:
        tissue_id = int(args.tissue_id)
    else:
        if tissue_name not in TISSUE_TO_IDX:
            raise ValueError(f"Unknown tissue '{args.tissue}'. Available: {sorted(TISSUE_TO_IDX)}")
        tissue_id = TISSUE_TO_IDX[tissue_name]

    for key in args.keys:
        try:
            img_u8_224 = reader.get(key)  # (224,224,3) uint8
        except KeyError as e:
            print(f"[MISS] {e}")
            continue

        x = to_tensor(Image.fromarray(img_u8_224).convert("RGB"))
        x384 = resize_224_to_384(x)
        x384 = normalize(x384).unsqueeze(0).to(device)

        cam224, pred_idx, pred_prob = cammer(x384, tissue_id=tissue_id, target_index=None)
        pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
        true_label = ann.get(key, "(unknown)")

        ovly_png = os.path.join(out_dir, f"{key}_overlay.png")
        save_overlay(img_u8_224, cam224, ovly_png, alpha=0.45)

        print(f"[OK] key={key}  pred={pred_name}  prob={pred_prob:.3f}  true={true_label}")
        print(f"     -> overlay: {ovly_png}")

    reader.close()
    print(f"Done. Outputs in: {out_dir}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
