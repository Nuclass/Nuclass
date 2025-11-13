"""
Safe fallbacks for torchvision optional operators.

Some lightweight Conda builds of torchvision omit C++ ops such as NMS, which
causes downstream imports (e.g., timm) to fail during registration. We install
a minimal pure-PyTorch implementation on demand so the rest of the codebase can
remain unchanged while still nudging users to install the full build when
possible.
"""

from __future__ import annotations

import warnings

import torch
from torch.library import Library

_PATCHED = False


def _box_areas(boxes: torch.Tensor) -> torch.Tensor:
    """Compute axis-aligned box areas for [..., 4] tensors."""
    widths = (boxes[..., 2] - boxes[..., 0]).clamp_min_(0)
    heights = (boxes[..., 3] - boxes[..., 1]).clamp_min_(0)
    return widths * heights


def _pairwise_iou(anchor: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """IoU between a single anchor box and a set of boxes."""
    xx1 = torch.maximum(anchor[0], others[:, 0])
    yy1 = torch.maximum(anchor[1], others[:, 1])
    xx2 = torch.minimum(anchor[2], others[:, 2])
    yy2 = torch.minimum(anchor[3], others[:, 3])

    inter = (xx2 - xx1).clamp_min_(0) * (yy2 - yy1).clamp_min_(0)
    union = _box_areas(anchor.unsqueeze(0)) + _box_areas(others) - inter
    iou = inter / torch.clamp(union, min=1e-7)
    return iou


def _pure_torch_nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Vanilla NMS with the same signature as torchvision.ops.nms."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    idxs = scores.argsort(descending=True)
    keep: list[int] = []

    while idxs.numel() > 0:
        anchor_idx = idxs[0]
        keep.append(int(anchor_idx))
        if idxs.numel() == 1:
            break

        anchor_box = boxes[anchor_idx]
        remaining = idxs[1:]
        ious = _pairwise_iou(anchor_box, boxes[remaining])
        idxs = remaining[ious <= threshold]

    return boxes.new_tensor(keep, dtype=torch.long)


def ensure_torchvision_nms() -> None:
    """
    Ensure torchvision::nms exists even on stripped builds.

    The fallback is slower but keeps notebooks/tutorials functional. We only
    install it if the operator is truly missing.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        getattr(torch.ops.torchvision, "nms")
        return
    except (RuntimeError, AttributeError):
        pass

    def _nms_stub(dets: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        if dets.dim() != 2 or dets.size(1) != 4:
            raise RuntimeError("Expected `dets` to have shape [N, 4].")
        if scores.dim() != 1 or scores.numel() != dets.size(0):
            raise RuntimeError("Scores must be 1D with the same length as boxes.")
        return _pure_torch_nms(dets, scores, float(iou_threshold))

    lib = Library("torchvision", "DEF")
    lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    lib.impl("nms", _nms_stub, "CompositeExplicitAutograd")

    warnings.warn(
        "torchvision::nms operator unavailable; using a pure-PyTorch fallback. "
        "Install torchvision with compiled ops for better performance.",
        RuntimeWarning,
    )
    _PATCHED = True
