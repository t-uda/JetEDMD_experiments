"""テスト共通の小さなユーティリティ"""

from __future__ import annotations

import numpy as np


def linear_trajectory(
    decay: float = -1.0, t_end: float = 2.0, n: int = 201, dims: int = 1
):
    t = np.linspace(0.0, t_end, n)
    base = np.exp(decay * t)
    y = np.column_stack([base ** (k + 1) for k in range(dims)])
    return t, y


def discrete_trajectory(
    factors: tuple[float, ...] = (0.9,), dt: float = 0.1, n: int = 120
):
    t = np.arange(n, dtype=float) * dt
    powers = np.arange(n)
    y = np.column_stack([f**powers for f in factors])
    return t, y


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return np.linalg.norm(a)
    return np.linalg.norm(a - b) / denom
