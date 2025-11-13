# -*- coding: utf-8 -*-
"""
Grad-CAM for LOKI (open_clip CoCa ViT-L/14 + Linear Head).
Provide the H5 path, model checkpoint, and the keys you want to visualize.
"""

import os, warnings, argparse
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from open_clip import create_model_from_pretrained, create_model

# Backward-compatibility for older open_clip versions that ignore weights_only
try:
    from open_clip.model import CoCa
    _orig_init = CoCa.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs.pop("weights_only", None)
        return _orig_init(self, *args, **kwargs)
    CoCa.__init__ = _patched_init
except Exception:
    pass

# Default output directory (relative to the working directory)
DEFAULT_OUT_DIR = Path("results") / "figure_reproduce" / "grad_cam" / "loki"
DEFAULT_KEYS = ["gidojinm-1","ignfhjao-1","ijjnlofi-1","lldfeija-1","gidojinm-1","oiimjodd-1"]

# Optional CoCa pretrained weights passed via --pretrained
LOCAL_COCA_PRETRAIN: Optional[str] = None
MODEL_NAME = "coca_ViT-L-14"

# CLIP/CoCa normalization
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

# ----- Model (structure matches training ckpt) -----
def _canon_key(k: str) -> str:
    for p in ("module.", "model.", "net.", "pl_module."):
        if k.startswith(p):
            return k[len(p):]
    return k

class LOKIGradModel(nn.Module):
    """open_clip CoCa ViT-L/14 backbone + linear classifier sized by ckpt."""
    def __init__(self, backbone, head_in_dim: int, head_out_dim: int):
        super().__init__()
        self.backbone = backbone
        # Keep gradients for CAM
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.norm = nn.LayerNorm(int(head_in_dim))
        self.head = nn.Linear(int(head_in_dim), int(head_out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.encode_image(x)   # [B, D]
        feats = self.norm(feats)
        return self.head(feats)

def _make_backbone(device: torch.device, pretrained_path: Optional[str]):
    # Prefer the training-time pretrained weights; otherwise fall back to an uninitialized backbone
    try:
        if pretrained_path and os.path.exists(pretrained_path):
            bb, _ = create_model_from_pretrained(MODEL_NAME, device="cpu",
                                                 pretrained=pretrained_path,
                                                 weights_only=False)
        else:
            raise RuntimeError("no local pretrained provided, fallback")
    except Exception:
        bb = create_model(MODEL_NAME, pretrained=False, precision="fp32")
    bb.to(device).eval()
    return bb

def build_model_from_loki_ckpt(ckpt_path: str, device: torch.device) -> Tuple[LOKIGradModel, List[str]]:
    obj = torch.load(ckpt_path, map_location="cpu")
    sd_all = obj.get("state_dict", obj)
    sd_all = { _canon_key(k): v for k, v in sd_all.items() }
    hparams = obj.get("hyper_parameters", {})

    # Infer head shape from checkpoint weights
    if "head.weight" not in sd_all or sd_all["head.weight"].ndim != 2:
        raise RuntimeError("head.weight missing in checkpoint; cannot infer classifier shape.")
    head_out_dim = int(sd_all["head.weight"].shape[0])
    head_in_dim  = int(sd_all["head.weight"].shape[1])
    print(f"[ckpt] head: out={head_out_dim}  in={head_in_dim}")

    # Derive class names or use placeholders
    class_names = hparams.get("class_names", None)
    if (class_names is None) or (len(class_names) != head_out_dim):
        class_names = [f"cls_{i}" for i in range(head_out_dim)]

    # Build backbone + inference model
    backbone = _make_backbone(device, pretrained_path=LOCAL_COCA_PRETRAIN)
    model = LOKIGradModel(backbone, head_in_dim=head_in_dim, head_out_dim=head_out_dim).to(device).eval()

    # Load only backbone.*, norm.*, head.*
    new_sd = {k: v for k, v in sd_all.items() if k.startswith(("backbone.", "norm.", "head."))}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)} (first 6): {missing[:6]}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)} (first 6): {unexpected[:6]}")

    return model, class_names

# ----- Grad-CAM (first Conv2d in visual tower) -----
class GradCAMonFirstConv:
    """Target layer: patch embedding Conv2d. CAM = ReLU(sum_k(mean_{HW}(dY/dA_k) * A_k))."""
    def __init__(self, visual_module: nn.Module, device: torch.device):
        self.visual = visual_module
        self.device = device
        self.target_layer = self._find_first_conv()
        if self.target_layer is None:
            raise RuntimeError("Could not locate a Conv2d patch embed layer for Grad-CAM.")
        self._f = None
        self._handles = []

    def _find_first_conv(self):
        # Try common attribute paths
        paths = [
            "trunk.patch_embed.proj",
            "patch_embed.proj",
            "conv1",
        ]
        for p in paths:
            obj = self.visual
            ok = True
            for attr in p.split("."):
                if not hasattr(obj, attr):
                    ok = False; break
                obj = getattr(obj, attr)
            if ok and isinstance(obj, nn.Conv2d):
                return obj
        # Fallback: first Conv2d under the vision tower (prefer kernel size 14)
        pick, pick_ks = None, None
        for m in self.visual.modules():
            if isinstance(m, nn.Conv2d):
                ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
                if ks == (14, 14):
                    return m
                if pick is None:
                    pick, pick_ks = m, ks
        return pick

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

    def __call__(self, model: nn.Module, x224_norm: torch.Tensor, target_index: int = None):
        """
        x224_norm: [1,3,224,224] after CLIP/CoCa normalization
        return: cam_224, pred_idx, pred_prob
        """
        self._register()
        try:
            model.zero_grad(set_to_none=True)
            logits = model(x224_norm)                      # [1, K]
            probs  = torch.softmax(logits, dim=1)
            pred_idx = int(probs[0].argmax().item()) if target_index is None else int(target_index)
            pred_prob = float(probs[0, pred_idx].item())

            # Backprop to target layer
            if self._f is not None and self._f.grad is not None:
                self._f.grad.zero_()
            logits[0, pred_idx].backward(retain_graph=False)

            feat = self._f.detach()                        # [1, C, H', W']
            grad = self._f.grad
            if grad is None:
                raise RuntimeError("Missing gradients for the target layer (check requires_grad/eval state).")

            weights = torch.mean(grad, dim=(2,3), keepdim=True)   # [1, C, 1, 1]
            cam = torch.sum(weights * feat, dim=1, keepdim=True)  # [1, 1, H', W']
            cam = torch.relu(cam)

            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-6)
            cam_224 = self._to_224(cam)
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
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for LOKI (CoCa ViT-L/14).")
    parser.add_argument("--h5-path", required=True, help="Path to patches_224x224.h5")
    parser.add_argument("--ckpt-path", required=True, help="Path to the LOKI checkpoint (.ckpt)")
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, help="Cell keys to visualize")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Directory to store overlays/PNGs")
    parser.add_argument("--pretrained", default=None, help="Optional CoCa pretrained weights for backbone")
    parser.add_argument("--device", choices=["auto","cuda","cpu"], default="auto", help="Force cpu/cuda or auto-detect")
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def main():
    global LOCAL_COCA_PRETRAIN
    args = parse_args()
    LOCAL_COCA_PRETRAIN = args.pretrained

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

    # Build model and load checkpoint
    model, class_names = build_model_from_loki_ckpt(args.ckpt_path, device=device)
    print(f"[ckpt] LOKI ready, classes={len(class_names)}")

    # CLIP/CoCa preprocessing
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    reader = H5Reader(args.h5_path)
    cammer = GradCAMonFirstConv(model.backbone.visual, device=device)

    for key in args.keys:
        try:
            img_u8_224 = reader.get(key)  # (224,224,3) uint8
        except KeyError as e:
            print(f"[MISS] {e}")
            continue

        x = preprocess(Image.fromarray(img_u8_224).convert("RGB")).unsqueeze(0).to(device)  # [1,3,224,224]

        cam224, pred_idx, pred_prob = cammer(model, x, target_index=None)
        pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)

        ovly_png = os.path.join(out_dir, f"{key}_overlay.png")
        save_overlay(img_u8_224, cam224, ovly_png, alpha=0.45)

        print(f"[OK] key={key}  pred={pred_name}  prob={pred_prob:.3f}  -> {ovly_png}")

    reader.close()
    print(f"Done. Outputs in: {out_dir}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
