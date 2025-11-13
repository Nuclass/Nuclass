# -*- coding: utf-8 -*-
"""
Grad-CAM for PLIP finetune (CLIP ViT-L/14 + linear head).
Launch with your own `--h5-path`, `--ckpt-path`, and list of cell `--keys`
to produce overlay PNGs for the selected patches.
"""

import os, json, warnings, argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as T

from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD  # Placeholder import; unused
from transformers import CLIPModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- Config -----
DEFAULT_OUT_DIR = Path("results") / "figure_reproduce" / "grad_cam" / "plip"
DEFAULT_KEYS = ["gidnfkoa-1", "dicdigfp-1", "mlabfdco-1", "nojdmjic-1" , "ignfhjao-1", "einlicic-1"]

# CLIP preprocessing stats used during finetuning
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

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

# ----- Model (PLIP inference + linear head) -----
class PLIPInfer(nn.Module):
    """Mirror training inference: features = clip.get_image_features -> LN -> linear head."""
    def __init__(self, model_name_or_path: str, head_in_dim: int, head_out_dim: int, local_only: bool=False):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name_or_path, local_files_only=bool(local_only))
        # Keep gradients on the CLIP tower for CAM
        for p in self.clip.parameters():
            p.requires_grad = True
        self.norm = nn.LayerNorm(int(head_in_dim))
        self.head = nn.Linear(int(head_in_dim), int(head_out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.clip.get_image_features(pixel_values=x)  # [B, D]
        feats = self.norm(feats)
        logits = self.head(feats)
        return logits

def _canon_key(k: str) -> str:
    # Strip common Lightning/DP prefixes
    for p in ("module.", "model.", "net.", "pl_module."):
        if k.startswith(p):
            return k[len(p):]
    return k

def build_model_from_plip_ckpt(ckpt_path: str, device: torch.device) -> Tuple[PLIPInfer, List[str]]:
    obj = torch.load(ckpt_path, map_location="cpu")
    sd_all = obj.get("state_dict", obj)
    hparams = obj.get("hyper_parameters", {})

    # Canonicalize keys
    sd_all = { _canon_key(k): v for k, v in sd_all.items() }

    # Infer head shape directly from checkpoint weights
    if "head.weight" not in sd_all or sd_all["head.weight"].ndim != 2:
        raise RuntimeError("head.weight missing (or malformed) in checkpoint; cannot build classifier head.")
    head_out_dim = int(sd_all["head.weight"].shape[0])  # num_classes
    head_in_dim  = int(sd_all["head.weight"].shape[1])  # feature dim

    # Derive class names or fallback to placeholders
    class_names = hparams.get("class_names", None)
    if (class_names is None) or (len(class_names) != head_out_dim):
        class_names = [f"cls_{i}" for i in range(head_out_dim)]

    print(f"[ckpt] head: out={head_out_dim}  in={head_in_dim}")

    # Build inference model on vinid/plip backbone
    model = PLIPInfer("vinid/plip", head_in_dim=head_in_dim, head_out_dim=head_out_dim, local_only=False)

    # Load clip/norm/head weights only
    loadable = {}
    for k, v in sd_all.items():
        if k.startswith(("clip.", "norm.", "head.")):
            loadable[k] = v
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")

    model.to(device).eval()
    return model, class_names

# ----- Grad-CAM (patch embedding conv) -----
class GradCAMonPatchConv:
    """Target layer: clip.vision_model.vision_model.embeddings.patch_embedding.projection (or equivalent).
    CAM = ReLU(sum_k(mean_{HW}(dY/dA_k) * A_k))"""
    def __init__(self, clip_model: CLIPModel, device: torch.device):
        self.clip = clip_model
        self.device = device
        self.target_layer = self._find_patch_conv()
        if self.target_layer is None:
            raise RuntimeError("Could not locate patch embedding Conv2d inside CLIP vision tower.")
        self._f = None
        self._handles = []

    def _find_patch_conv(self):
        # Try common attribute paths
        cand_attrs = [
            "vision_model.vision_model.embeddings.patch_embedding.projection",
            "vision_model.embeddings.patch_embedding.projection",
        ]
        for path in cand_attrs:
            obj = self.clip
            ok = True
            for attr in path.split("."):
                if not hasattr(obj, attr):
                    ok = False; break
                obj = getattr(obj, attr)
            if ok and isinstance(obj, nn.Conv2d):
                return obj
        # Fallback: first Conv2d under vision_model (prefer kernel=14)
        vm = getattr(self.clip, "vision_model", None)
        best = None
        for m in vm.modules():
            if isinstance(m, nn.Conv2d):
                best = m
                if m.kernel_size == (14, 14):
                    return m
        return best

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
    def _to_224(self, cam: torch.Tensor) -> np.ndarray:
        cam = torch.nn.functional.interpolate(cam, size=(224,224), mode="bilinear", align_corners=False)
        cam = cam[0,0]
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-6)
        return cam.detach().cpu().numpy().astype(np.float32)

    def __call__(self, x224_norm: torch.Tensor, target_index: int = None):
        """
        x224_norm: [1,3,224,224] already CLIP-normalized
        return: cam_224 (H=W=224, float32 [0,1]), pred_idx, pred_prob
        """
        self._register()
        try:
            logits = self.clip.get_image_features(pixel_values=x224_norm)  # [1, D]
            # The classifier lives outside this helper; gradients should flow through the full model.
            # Use the wrapper below so logits originate from the linear head before backward.
            raise RuntimeError("Call this class through the wrapper that includes the linear head.")
        finally:
            self._remove()

# Grad-CAM wrapper that backpropagates through the linear head
class PLIPGradCAM:
    def __init__(self, model: PLIPInfer, device: torch.device):
        self.model = model
        self.device = device
        self.cammer = GradCAMonPatchConv(model.clip, device)

    def __call__(self, x224_norm: torch.Tensor, target_index: int = None):
        """
        x224_norm: [1,3,224,224]
        return: cam_224 (H=W=224, float32 [0,1]), pred_idx, pred_prob
        """
        # Register hooks
        self.cammer._register()
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x224_norm)                 # [1, K]
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(probs[0].argmax().item()) if target_index is None else int(target_index)
            pred_prob = float(probs[0, pred_idx].item())

            # Backprop to target layer
            if self.cammer._f is not None and self.cammer._f.grad is not None:
                self.cammer._f.grad.zero_()
            logits[0, pred_idx].backward(retain_graph=False)

            feat = self.cammer._f.detach()                 # [1, C, H', W']
            grad = self.cammer._f.grad
            if grad is None:
                raise RuntimeError("Gradients for the target layer are missing (check requires_grad/eval state).")

            weights = torch.mean(grad, dim=(2,3), keepdim=True)   # [1, C, 1, 1]
            cam = torch.sum(weights * feat, dim=1, keepdim=True)  # [1, 1, H', W']
            cam = torch.relu(cam)

            # Normalize and resize to 224
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-6)
            cam_224 = self.cammer._to_224(cam)
            return cam_224, pred_idx, pred_prob
        finally:
            self.cammer._remove()

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
    parser = argparse.ArgumentParser(description="Grad-CAM for PLIP finetune (CLIP ViT-L/14).")
    parser.add_argument("--h5-path", required=True, help="Path to patches_224x224.h5")
    parser.add_argument("--ckpt-path", required=True, help="Path to PLIP checkpoint (.ckpt)")
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, help="Cell keys to visualize")
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

    model, class_names = build_model_from_plip_ckpt(args.ckpt_path, device=device)
    print(f"[ckpt] PLIP ready, classes={len(class_names)}")

    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    reader = H5Reader(args.h5_path)
    cammer = PLIPGradCAM(model, device=device)

    for key in args.keys:
        try:
            img_u8_224 = reader.get(key)
        except KeyError as e:
            print(f"[MISS] {e}")
            continue

        x = preprocess(Image.fromarray(img_u8_224).convert("RGB")).unsqueeze(0).to(device)

        cam224, pred_idx, pred_prob = cammer(x, target_index=None)
        pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)

        ovly_png = os.path.join(out_dir, f"{key}_overlay.png")
        save_overlay(img_u8_224, cam224, ovly_png, alpha=0.45)

        print(f"[OK] key={key}  pred={pred_name}  prob={pred_prob:.3f}  -> {ovly_png}")

    reader.close()
    print(f"Done. Outputs in: {out_dir}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
