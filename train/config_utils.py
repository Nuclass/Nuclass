"""Tiny helpers that keep released configs environment-agnostic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

__all__ = [
    "resolve_data_dirs",
    "resolve_results_dir",
    "get_checkpoint_dir",
    "resolve_checkpoint_path",
]


def _expand(pathish: str | Path) -> Path:
    """Return an absolute path with user/home markers resolved."""
    return Path(pathish).expanduser().resolve()


def _ensure_dir(path: Path) -> Path:
    """Make sure a directory exists before using it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _split_paths(value: str) -> List[str]:
    chunks = [chunk.strip() for chunk in value.split(os.pathsep) if chunk.strip()]
    return [str(_expand(chunk)) for chunk in chunks]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_dirs(
    dataset_names: Sequence[str] | None = None,
    env_dirs: str = "NUCLASS_DATA_DIRS",
    env_root: str = "NUCLASS_DATA_ROOT",
) -> List[str]:
    """
    Build absolute dataset paths.
    Prefer NUCLASS_DATA_DIRS (os.pathsep separated); otherwise use NUCLASS_DATA_ROOT/<name>.
    """
    override = os.environ.get(env_dirs)
    if override:
        dirs = _split_paths(override)
        if not dirs:
            raise RuntimeError(f"{env_dirs} is set but empty.")
        return dirs

    data_root = os.environ.get(env_root)
    if not data_root:
        raise RuntimeError(
            f"Set {env_dirs} (preferred) or {env_root} so the training scripts know where the datasets live."
        )
    if not dataset_names:
        raise RuntimeError(
            "Provide dataset_names to resolve_data_dirs() when relying on NUCLASS_DATA_ROOT."
        )
    base = _expand(data_root)
    return [str(_expand(base / name)) for name in dataset_names]


def resolve_results_dir(
    experiment_name: str,
    env_key: str = "NUCLASS_RESULTS_ROOT",
) -> str:
    """Return the directory used for inference dumps and plots."""
    if not experiment_name:
        raise ValueError("experiment_name is required to resolve a results path.")
    root = _expand(os.environ.get(env_key, _repo_root() / "results"))
    return str(_ensure_dir(root / experiment_name))


def get_checkpoint_dir(
    experiment_name: str,
    env_key: str = "NUCLASS_CHECKPOINT_ROOT",
) -> str:
    """Directory where PyTorch Lightning checkpoints should be stored."""
    if not experiment_name:
        raise ValueError("experiment_name is required to resolve a checkpoint path.")
    root = _expand(os.environ.get(env_key, _repo_root() / "checkpoints"))
    return str(_ensure_dir(root / experiment_name))


def resolve_checkpoint_path(
    relative_path: str,
    env_key: str = "NUCLASS_CHECKPOINT_ROOT",
) -> str:
    """Resolve a released checkpoint inside the checkpoint root."""
    if not relative_path:
        raise ValueError("relative_path must not be empty.")
    root = _expand(os.environ.get(env_key, _repo_root() / "checkpoints"))
    return str(_expand(root / relative_path))
