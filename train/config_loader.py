"""Reusable helpers for YAML-based experiment configs with CLI overrides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml

def _build_parser(description: str, default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help=f"Path to YAML config file (default: {default_config})",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config field using dot notation, e.g. train.lr=1e-4. VALUE is parsed as YAML.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the merged config and exit.",
    )
    return parser


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping/object.")
    return data


def _set_by_path(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = [k for k in dotted_key.split(".") if k]
    if not keys:
        raise ValueError(f"Invalid override key '{dotted_key}'")
    target = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def _apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in KEY=VALUE form.")
        key, raw_value = item.split("=", 1)
        parsed_value = yaml.safe_load(raw_value)
        _set_by_path(config, key.strip(), parsed_value)


def load_experiment_config(
    default_config_path: Path,
    description: str,
) -> Tuple[Dict[str, Any], argparse.Namespace]:
    """
    Parse CLI arguments, load the YAML config, apply overrides, and return (config, args).
    """
    default_config_path = default_config_path.resolve()
    parser = _build_parser(description, default_config_path)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = _load_yaml(config_path)
    config["_config_file"] = str(config_path)
    _apply_overrides(config, args.override)

    if args.print_config:
        yaml.safe_dump(config, stream=sys.stdout, sort_keys=False)
        raise SystemExit(0)

    return config, args

