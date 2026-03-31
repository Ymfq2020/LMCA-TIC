"""Random seed management."""

from __future__ import annotations

import os
import random

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
