"""Learning-rate schedulers used by the LMCA-TIC trainer."""

from __future__ import annotations

try:
    from torch.optim.lr_scheduler import LambdaLR
except ImportError:  # pragma: no cover - optional dependency
    LambdaLR = None


def build_warmup_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    if LambdaLR is None:
        return None
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
