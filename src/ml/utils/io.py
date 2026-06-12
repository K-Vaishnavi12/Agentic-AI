"""IO helpers — JSON, joblib, torch state. All paths-agnostic, atomic where it matters."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any


def write_json(path: str | Path, obj: Any) -> Path:
    """Atomic JSON write. Creates parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_joblib(obj: Any, path: str | Path) -> Path:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_joblib(path: str | Path) -> Any:
    import joblib
    return joblib.load(path)


def save_torch(state_dict: dict, path: str | Path) -> Path:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    return path


def load_torch(path: str | Path, map_location: str = "cpu"):
    import torch
    return torch.load(path, map_location=map_location, weights_only=False)
