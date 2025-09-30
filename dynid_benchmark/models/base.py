from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class Model(ABC):
    """動的モデルの共通インターフェース"""

    name: str = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, t: np.ndarray, y: np.ndarray, u: Optional[np.ndarray] = None):
        pass

    @abstractmethod
    def predict_derivative(
        self, t: float, x: np.ndarray, u: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pass

    def rollout(
        self, t: np.ndarray, x0: np.ndarray, u: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # predict_derivative で定義されるベクトル場を時間グリッド上で RK4 積分
        def f(ti, xi, ui):
            return self.predict_derivative(ti, xi, ui)

        x = np.zeros((len(t), len(x0)), dtype=float)
        x[0] = x0.copy()
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            ui = None if u is None else u[i - 1]
            k1 = f(t[i - 1], x[i - 1], ui)
            k2 = f(t[i - 1] + 0.5 * dt, x[i - 1] + 0.5 * dt * k1, ui)
            k3 = f(t[i - 1] + 0.5 * dt, x[i - 1] + 0.5 * dt * k2, ui)
            k4 = f(t[i - 1] + dt, x[i - 1] + dt * k3, ui)
            x[i] = x[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x


class ZeroModel(Model):
    name = "zero"

    def fit(self, t, y, u=None):
        pass

    def predict_derivative(self, t, x, u=None):
        return 0.0 * x


# Registry
MODEL_REGISTRY = {"zero": ZeroModel}


def register_model(cls):
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError("register_model requires classes to define a non-empty 'name'")
    existing = MODEL_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Model '{name}' already registered by {existing.__module__}.{existing.__name__}"
        )
    MODEL_REGISTRY[name] = cls
    return cls
