import numpy as np
from typing import Dict, Optional, Callable
import json, os


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def vector_field_mse(
    f_true: Callable[[float, np.ndarray], np.ndarray],
    f_hat: Callable[[float, np.ndarray], np.ndarray],
    t_grid: np.ndarray,
    x_grid: np.ndarray,
) -> float:
    errs = []
    for i in range(len(t_grid)):
        errs.append(
            np.linalg.norm(f_hat(t_grid[i], x_grid[i]) - f_true(t_grid[i], x_grid[i]))
            ** 2
        )
    return float(np.mean(errs))


def rollout_rmse(t, x_true, x_hat):
    return rmse(x_true, x_hat)


def energy_drift(k, m, x, v):
    E = 0.5 * k * x**2 + 0.5 * m * v**2
    return float(E[-1] - E[0])


def save_metrics(path, metrics: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
