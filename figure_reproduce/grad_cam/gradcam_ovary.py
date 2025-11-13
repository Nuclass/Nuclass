# -*- coding: utf-8 -*-
"""
Grad-CAM for Path-A (UNI2-h + FiLM + MLP head).
Provide the annotations JSON, H5 path, checkpoint, and list of keys you want to visualize.
"""

import os, json, warnings, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from Nuclass.utils.torchvision_compat import ensure_torchvision_nms

ensure_torchvision_nms()
import torchvision.transforms as T
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Config -----
DEFAULT_OUT_DIR = Path("results") / "figure_reproduce" / "grad_cam" / "path_a"
DEFAULT_KEYS = ["gidnfkoa-1", "dicdigfp-1", "mlabfdco-1", "nojdmjic-1" , "ignfhjao-1", "einlicic-1"]

# Tissue IDs (match training order: alphabetical)
TISSUE_TO_IDX = {"breast":0,"colon":1,"heart":2,"kidney":3,"liver":4,"lung":5,"ovary":6,"pancreas":7}

# ImageNet normalization (same as UNI2-h reference)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

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
    """Return {cell_id -> label} if label columns exist."""
    if not os.path.exists(json_path):
        return {}
    df = pd.read_json(json_path, orient="index").reset_index().rename(columns={"index":"cell_id"})
    # Pick the first available label column
    label_col = None
    for k in ["class_id","label","cell_type","type","annotation"]:
        if k in df.columns:
            label_col = k
            break
    if label_col is None:
        return {}
    df["cell_id"] = df["cell_id"].astype(str)
    return dict(zip(df["cell_id"], df[label_col].astype(str)))

# ----- Path-A model -----
class FiLM(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int):
        super().__init__()
        hid = max(64, in_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.SiLU(),
            nn.Linear(hid, 2 * feat_dim)
        )
    def forward(self, t):
        gamma, beta = self.net(t).chunk(2, dim=-1)
        return gamma, beta

class PathA_UNI2h(nn.Module):
    """Inference-only stack: UNI2-h backbone -> LayerNorm -> FiLM(tissue) -> head.
    Uses pretrained=False to rely solely on checkpoint weights."""
    def __init__(self, num_classes: int, num_tissues: int, tissue_dim: int = 64):
        super().__init__()
        uni_cfg = {
            'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
            'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
            'num_classes': 0, 'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
            'reg_tokens': 8, 'dynamic_img_size': True
        }
        self.uni_encoder = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=False, **uni_cfg)
        self.uni_feature_dim = self.uni_encoder.embed_dim

        self.tissue_embedder = nn.Embedding(num_tissues, tissue_dim)
        self.tissue_film     = FiLM(tissue_dim, self.uni_feature_dim)
        self.feat_norm       = nn.LayerNorm(self.uni_feature_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.uni_feature_dim, 1024), nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    def extract_features(self, x: torch.Tensor, tissues: torch.Tensor) -> torch.Tensor:
        # UNI2-h forward -> [B,1536]
        feat = self.uni_encoder(x)
        feat = self.feat_norm(feat)
        t = self.tissue_embedder(tissues)
        gamma, beta = self.tissue_film(t)
        feat = feat * (1 + gamma) + beta
        return feat

    def forward(self, x: torch.Tensor, tissues: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x, tissues))

def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[PathA_UNI2h, List[str]]:
    obj = torch.load(ckpt_path, map_location="cpu")
    hparams = obj.get("hyper_parameters", {})
    class_names = hparams.get("class_names", None)
    if class_names is None:
        # Fallback: infer from final linear head weight
        sd = obj.get("state_dict", obj)
        last_w = None
        # Use the largest ".head." weight matrix
        for k,v in sd.items():
            if k.endswith(".weight") and ".head." in k and v.ndim==2:
                if (last_w is None) or (v.shape[0] > last_w.shape[0]):
                    last_w = v
        if last_w is None:
            raise RuntimeError("Unable to infer num_classes (missing class_names and head.*.weight).")
        num_classes = int(last_w.shape[0])
        class_names = [f"cls_{i}" for i in range(num_classes)]
    num_classes = len(class_names)

    num_tissues = int(hparams.get("num_tissues", 8))
    model = PathA_UNI2h(num_classes=num_classes, num_tissues=num_tissues, tissue_dim=int(hparams.get("tissue_embedding_dim", 64)))
    sd = obj.get("state_dict", obj)

    # Load inference-related keys only
    new = {}
    for k,v in sd.items():
        if k.startswith(("uni_encoder.","tissue_embedder.","tissue_film.","feat_norm.","head.")):
            new[k] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)} (show first 8) {missing[:8]}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)} (show first 8) {unexpected[:8]}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = True     # Grad-CAM requires gradients
    return model, class_names

# --------------------------- Grad-CAM (hook patch_embed.proj) ---------------------------
class ViTGradCAMonPatchEmbed:
    """Standard Grad-CAM on the ViT patch embedding conv:
      cam = ReLU(sum_k(mean_{HW}(dY/dA_k) * A_k))."""
    def __init__(self, model: PathA_UNI2h, device: torch.device):
        self.model = model
        self.device = device
        # Locate target layer: prefer uni_encoder.patch_embed.proj
        target = None
        try:
            target = model.uni_encoder.patch_embed.proj
        except Exception:
            pass
        if target is None:
            # Fallback: first Conv2d inside uni_encoder
            for m in model.uni_encoder.modules():
                if isinstance(m, nn.Conv2d):
                    target = m
                    break
        if target is None:
            raise RuntimeError("Could not find a ViT patch embedding Conv2d for Grad-CAM.")
        self.target_layer = target

        self._f = None
        self._handles = []

    def _register(self):
        def fwd_hook(_m, _in, out):
            # out: [B, C, H', W']; retain gradients
            self._f = out
            try:
                self._f.retain_grad()
            except Exception:
                pass

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))

    def _remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def __call__(self, x: torch.Tensor, tissue_id: int, target_index: int = None, class_names: List[str] = None):
        """
        x: [1,3,224,224] already ImageNet-normalized
        tissue_id: int
        target_index: explicit class index; defaults to argmax
        return: cam_224 (H=W=224, float32 [0,1]), pred_idx, pred_prob
        """
        self._register()
        try:
            tissues = torch.tensor([tissue_id], dtype=torch.long, device=self.device)
            logits = self.model(x, tissues)                  # Hook captures patch_embed.proj output self._f
            probs  = torch.softmax(logits, dim=1)
            pred_prob, pred_idx = float(probs[0].max().item()), int(probs[0].argmax().item())

            if target_index is None:
                target_index = pred_idx

            score = logits[0, target_index]
            self.model.zero_grad(set_to_none=True)
            if self._f is not None and self._f.grad is not None:
                self._f.grad.zero_()

            # Backprop to obtain dY/dA
            score.backward(retain_graph=False)

            # Extract activations and gradients
            feat = self._f.detach()                        # [1, C, H', W']
            grad = self._f.grad if (self._f is not None and self._f.grad is not None) else None
            if grad is None:
                raise RuntimeError("Missing gradients for the target layer (check requires_grad/eval state).")

            # Channel weights = spatial mean of gradients
            weights = torch.mean(grad, dim=(2,3), keepdim=True)   # [1, C, 1, 1]
            cam = torch.sum(weights * feat, dim=1, keepdim=True)  # [1, 1, H', W']
            cam = torch.relu(cam)
            # Normalize and upsample to 224
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-6)
            cam_up = torch.nn.functional.interpolate(cam, size=(224,224), mode="bilinear", align_corners=False)
            cam_np = cam_up[0,0].detach().cpu().numpy().astype(np.float32)
            return cam_np, pred_idx, pred_prob
        finally:
            self._remove()

# ----- Visualization -----
def save_overlay(img_u8: np.ndarray, cam_224: np.ndarray, out_png: str, alpha: float = 0.45):
    """
    img_u8: (224,224,3) uint8 source image
    cam_224: (224,224) float in [0,1]
    """
    plt.figure(figsize=(3,3), dpi=224/3)
    plt.axis("off")
    plt.imshow(img_u8)
    plt.imshow(cam_224, cmap="jet", alpha=alpha)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_heatmap(cam_224: np.ndarray, out_png: str):
    plt.figure(figsize=(3,3), dpi=224/3)
    plt.axis("off")
    plt.imshow(cam_224, cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close()

# ----- Main -----
def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM for Path-A (UNI2-h + FiLM).")
    parser.add_argument("--anno-json", required=True, help="Path to annotations.json")
    parser.add_argument("--h5-path", required=True, help="Path to patches_224x224.h5")
    parser.add_argument("--ckpt-path", required=True, help="Path to Path-A checkpoint (.ckpt)")
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
    model, class_names = build_model_from_ckpt(args.ckpt_path, device=device)
    print(f"[ckpt] classes={len(class_names)}")

    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    reader = H5Reader(args.h5_path)
    cammer = ViTGradCAMonPatchEmbed(model, device=device)

    tissue_name = args.tissue.lower()
    if args.tissue_id is not None:
        tissue_id = int(args.tissue_id)
    else:
        if tissue_name not in TISSUE_TO_IDX:
            raise ValueError(f"Unknown tissue '{args.tissue}'. Available: {sorted(TISSUE_TO_IDX)}")
        tissue_id = TISSUE_TO_IDX[tissue_name]

    for key in args.keys:
        try:
            img_u8 = reader.get(key)
        except KeyError as e:
            print(f"[MISS] {e}")
            continue

        x = to_tensor(Image.fromarray(img_u8).convert("RGB"))
        x = normalize(x).unsqueeze(0).to(device)

        cam224, pred_idx, pred_prob = cammer(x, tissue_id=tissue_id, target_index=None, class_names=class_names)
        pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
        true_label = ann.get(key, "(unknown)")

        heat_png = os.path.join(out_dir, f"{key}_cam.png")
        ovly_png = os.path.join(out_dir, f"{key}_overlay.png")
        save_heatmap(cam224, heat_png)
        save_overlay(img_u8, cam224, ovly_png, alpha=0.45)

        print(f"[OK] key={key}  pred={pred_name}  prob={pred_prob:.3f}  true={true_label}")
        print(f"     -> heat: {heat_png}")
        print(f"     -> ovly: {ovly_png}")

    reader.close()
    print(f"Done. Outputs in: {out_dir}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
