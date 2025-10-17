"""PyKoopman を dynid_benchmark モデル API にラップするアダプタ"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Model, register_model

try:  # pragma: no cover - optional dependency guard
    from pykoopman import Koopman
    from pykoopman.regression import EDMD, EDMDc
    from pykoopman.observables import Polynomial

    _PYKOOPMAN_OK = True
except Exception as err:  # pragma: no cover - import guard
    _PYKOOPMAN_OK = False
    _PYKOOPMAN_ERR = err
    Koopman = EDMD = EDMDc = Polynomial = None  # type: ignore


def _ensure_pykoopman_available() -> None:
    if not _PYKOOPMAN_OK:
        raise RuntimeError(
            "PyKoopman がインストールされていません。`poetry install` を実行してください。"
        ) from _PYKOOPMAN_ERR


def _ensure_uniform_timestep(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("時刻配列 t は一次元で長さ 2 以上である必要があります")
    dt = np.diff(t)
    dt0 = float(dt[0])
    if not np.allclose(dt, dt0, rtol=1e-6, atol=1e-12):
        raise ValueError("PyKoopman モデルは等間隔サンプリングを前提とします")
    return dt0


def _to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError("入力は 1 次元または 2 次元配列である必要があります")


@dataclass
class _KoopmanConfig:
    poly_order: int = 2
    include_bias: bool = True
    svd_rank: float = 1.0
    tlsq_rank: int = 0


class _PyKoopmanBase(Model):
    """PyKoopman を透過的に呼び出す基底クラス"""

    name: str = "pykoopman_base"
    with_control: bool = False

    def __init__(
        self,
        poly_order: int = 2,
        include_bias: bool = True,
        svd_rank: float = 1.0,
        tlsq_rank: int = 0,
    ) -> None:
        super().__init__(
            poly_order=poly_order,
            include_bias=include_bias,
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
        )
        self.cfg = _KoopmanConfig(
            poly_order=poly_order,
            include_bias=include_bias,
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
        )
        self._model: Optional[Koopman] = None
        self._dt: Optional[float] = None

    # -- helpers -----------------------------------------------------------------
    def _build_observables(self) -> Polynomial:
        return Polynomial(degree=self.cfg.poly_order, include_bias=self.cfg.include_bias)

    def _build_regressor(self):
        if self.with_control:
            return EDMDc()
        return EDMD(svd_rank=self.cfg.svd_rank, tlsq_rank=self.cfg.tlsq_rank)

    def _init_model(self) -> Koopman:
        return Koopman(
            observables=self._build_observables(),
            regressor=self._build_regressor(),
            quiet=True,
        )

    # -- Model API ----------------------------------------------------------------
    def fit(self, t: np.ndarray, y: np.ndarray, u: Optional[np.ndarray] = None):
        _ensure_pykoopman_available()
        dt = _ensure_uniform_timestep(t)
        X = _to_2d(y)
        if len(X) < 2:
            raise ValueError("学習には少なくとも 2 ステップ以上のデータが必要です")
        Xk = X[:-1]
        Xkp1 = X[1:]

        U = None
        if self.with_control:
            if u is None:
                raise ValueError("制御入力付きモデルには u が必須です")
            U = _to_2d(u)
            if len(U) != len(X):
                raise ValueError("制御入力 u の長さは状態系列と一致している必要があります")
            U = U[:-1]
        elif u is not None:
            # 制御なしモデルに制御が渡された場合は警告を出すより明示的に無視する
            U = None

        model = self._init_model()
        if U is None:
            model.fit(Xk, y=Xkp1, dt=dt)
        else:
            model.fit(Xk, y=Xkp1, u=U, dt=dt)
        self._model = model
        self._dt = dt

    def predict_derivative(  # pragma: no cover - 離散モデル
        self, t: float, x: np.ndarray, u: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # 離散モデルのためベクトル場は定義しない
        return np.zeros_like(x)

    def rollout(
        self, t: np.ndarray, x0: np.ndarray, u: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("モデルが学習されていません")
        t = np.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError("時刻配列 t は一次元を想定しています")
        if len(t) == 0:
            raise ValueError("rollout の時刻配列 t が空です")

        x0 = np.asarray(x0, dtype=float)
        if x0.ndim != 1:
            raise ValueError("初期状態 x0 は一次元ベクトルを想定しています")

        if len(t) == 1:
            return x0[None, :]

        steps = len(t) - 1
        control = None
        if self.with_control:
            if u is None:
                raise ValueError("制御付きモデルには rollout 時にも u が必要です")
            control = _to_2d(u)
            if len(control) != len(t):
                raise ValueError("制御入力 u の長さは時刻列と一致する必要があります")
            control = control[:-1]
        elif u is not None:
            control = None

        sim = self._model.simulate(x0, u=control, n_steps=steps)
        Y = np.vstack([x0, np.asarray(sim, dtype=float)])
        return Y


@register_model
class PyKoopmanEDMD(_PyKoopmanBase):
    name = "pykoopman_edmd"
    with_control = False


@register_model
class PyKoopmanEDMDc(_PyKoopmanBase):
    name = "pykoopman_edmdc"
    with_control = True
