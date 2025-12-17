from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds across libraries for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    """Clip gradients to improve training stability."""

    if max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)
