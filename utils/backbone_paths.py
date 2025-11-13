"""
Helpers for resolving heavyweight backbone paths without leaking internal directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _candidate_paths(model_id: str, repo_root: Path) -> Iterable[Path]:
    sanitized = model_id.replace("hf-hub:", "").strip("/")
    parts = Path(sanitized)
    env_val = os.environ.get("NUCLASS_DINO_MODEL")
    if env_val:
        env_path = Path(env_val).expanduser()
        if env_path.is_dir():
            yield env_path
    roots = [
        repo_root / "backbones" / parts,
        repo_root / "checkpoints" / "backbones" / parts,
    ]
    for root in roots:
        path = Path(root)
        if path.is_dir():
            yield path


def resolve_dino_model_path(model_id: str, repo_root: Path | str) -> str:
    """
    Return a local directory for the requested DINO model if it exists,
    otherwise fall back to the original identifier (HF Hub string).
    """
    repo_root = Path(repo_root)
    for cand in _candidate_paths(model_id, repo_root):
        cfg = cand / "config.json"
        if cfg.exists():
            return str(cand)
    return model_id
