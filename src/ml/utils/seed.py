"""Reproducibility — seed everything we touch (numpy, torch, python, env)."""
from __future__ import annotations
import os
import random


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs we use. Safe to call before any training run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism for forward/backward passes
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
