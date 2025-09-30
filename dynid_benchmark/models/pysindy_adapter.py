"""PySINDy をラップして既存インターフェースと接続するモデル定義"""

from __future__ import annotations

from typing import Optional, Tuple

import math
import numpy as np

from .base import Model, register_model
from .sindy_stlsq import finite_difference as _finite_difference

try:
    import pysindy as ps

    _PYSINDY_OK = True
except Exception as err:  # pragma: no cover - import guard
    _PYSINDY_OK = False
    _PYSINDY_ERR = err
    ps = None  # type: ignore[assignment]

try:  # SINDy-PI は cvxpy を要求
    import cvxpy as _cvxpy  # noqa: F401

    _CVXPY_OK = True
except Exception:  # pragma: no cover - optional dependency guard
    _CVXPY_OK = False


# NumPy 2.0 で np.math が削除されたため PySINDy 互換のため再導入
if getattr(np, "math", None) is None:
    np.math = math  # type: ignore[attr-defined]


def _ensure_pysindy_available():
    if not _PYSINDY_OK:
        raise RuntimeError(
            "PySINDy がインストールされていません。poetry install 実行後に再試行してください."
        ) from _PYSINDY_ERR


def _build_feature_library(
    poly_order: int, fourier_order: int, include_constant: bool = True
):
    """多項式と Fourier の組み合わせライブラリを構築する"""

    libs = []
    poly_lib = ps.PolynomialLibrary(degree=poly_order, include_bias=include_constant)
    libs.append(poly_lib)
    if fourier_order and fourier_order > 0:
        libs.append(ps.FourierLibrary(n_frequencies=fourier_order))
    if len(libs) == 1:
        return libs[0]
    return ps.GeneralizedLibrary(libs)


def _build_differentiation(method: str, kwargs: Optional[dict]):
    """文字列指定から PySINDy の微分器を生成"""

    kwargs = kwargs or {}
    method = method.lower()
    if method in {"finite_difference", "fd"}:
        return ps.FiniteDifference(**kwargs)
    if method in {"smoothed_fd", "smoothed_finite_difference"}:
        return ps.SmoothedFiniteDifference(**kwargs)
    if method in {"spline", "spline_filter"}:
        return ps.SplineDerivativeFilter(**kwargs)
    raise ValueError(f"未知の differentiation_method={method}")


def _build_optimizer(mode: str, name: Optional[str], kwargs: Optional[dict]):
    """SINDy/SINDy-PI それぞれに応じた最適化器を初期化"""

    kwargs = kwargs or {}
    if mode == "sindy_pi":
        if not _CVXPY_OK:
            raise RuntimeError(
                "PySINDy-PI を利用するには cvxpy のインストールが必要です。"
            )
        defaults = {"max_iter": 20}
        defaults.update(kwargs)
        optimizer_cls = getattr(ps, "SINDyPI", None)
        if optimizer_cls is None and hasattr(ps, "optimizers"):
            optimizer_cls = getattr(ps.optimizers, "SINDyPI", None)
        if optimizer_cls is None:
            raise AttributeError("PySINDy に SINDyPI optimizer が見つかりません")
        return optimizer_cls(**defaults)

    opt_name = (name or "stlsq").lower()
    if opt_name == "stlsq":
        defaults = {"threshold": 0.1}
        defaults.update(kwargs)
        return ps.STLSQ(**defaults)
    if opt_name == "sr3":
        defaults = {"threshold": 0.1, "nu": 1e-6}
        defaults.update(kwargs)
        return ps.SR3(**defaults)
    if opt_name == "l0":
        defaults = {"threshold": 0.1}
        defaults.update(kwargs)
        return ps.ConstrainedSR3(**defaults)
    raise ValueError(f"未知の optimizer={opt_name}")


def _prepare_time_arg(t: np.ndarray) -> Tuple[np.ndarray | float, float]:
    """PySINDy へ渡す t 引数（一定刻みの場合はスカラー）を整形"""

    t = np.asarray(t)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("時刻配列 t は長さ >=2 の一次元である必要があります")
    dt = np.diff(t)
    dt0 = float(dt[0])
    if np.allclose(dt, dt0, rtol=1e-6, atol=1e-12):
        return dt0, dt0
    return t, dt0


class _PySINDyAdapter(Model):
    """PySINDy を統一インターフェースにブリッジする基底クラス"""

    mode: str = "sindy"
    name: str = "pysindy"

    def __init__(
        self,
        poly_order: int = 3,
        fourier_order: int = 0,
        include_constant: bool = True,
        optimizer: Optional[str] = None,
        optimizer_kwargs: Optional[dict] = None,
        differentiation_method: str = "smoothed_fd",
        differentiation_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            poly_order=poly_order,
            fourier_order=fourier_order,
            include_constant=include_constant,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            differentiation_method=differentiation_method,
            differentiation_kwargs=differentiation_kwargs,
        )
        self.poly_order = poly_order
        self.fourier_order = fourier_order
        self.include_constant = include_constant
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.differentiation_method = differentiation_method
        self.differentiation_kwargs = differentiation_kwargs or {}
        self._model: Optional[ps.SINDy] = None
        self._dt: Optional[float] = None

    def _init_model(self) -> ps.SINDy:
        _ensure_pysindy_available()
        feature_library = _build_feature_library(
            self.poly_order, self.fourier_order, include_constant=self.include_constant
        )
        differentiation = _build_differentiation(
            self.differentiation_method, self.differentiation_kwargs
        )
        optimizer = _build_optimizer(self.mode, self.optimizer, self.optimizer_kwargs)
        return ps.SINDy(
            feature_library=feature_library,
            differentiation_method=differentiation,
            optimizer=optimizer,
            feature_names=None,
        )

    def fit(self, t: np.ndarray, y: np.ndarray, u: Optional[np.ndarray] = None):
        _ensure_pysindy_available()
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        control = None
        if u is not None:
            u = np.asarray(u, dtype=float)
            if u.ndim == 1:
                u = u[:, None]
            control = u
        t_arr = np.asarray(t, dtype=float)
        t_arg, dt_est = _prepare_time_arg(t_arr)
        self._dt = dt_est
        self._model = self._init_model()
        fit_kwargs = {"t": t_arg}
        if control is not None:
            fit_kwargs["control"] = control
        x_dot = _finite_difference(y, t_arr)
        self._model.fit(y, x_dot=x_dot, **fit_kwargs)

    def predict_derivative(
        self, t: float, x: np.ndarray, u: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("モデルが学習されていません")
        x = np.asarray(x, dtype=float)[None, :]
        if u is not None:
            control = np.asarray(u, dtype=float)[None, :]
            pred = self._model.predict(x, control=control)
        else:
            pred = self._model.predict(x)
        return np.asarray(pred[0], dtype=float)

    def rollout(
        self, t: np.ndarray, x0: np.ndarray, u: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("モデルが学習されていません")
        t = np.asarray(t, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        if t.ndim != 1:
            raise ValueError("rollout の時刻配列 t は一次元を想定しています")
        if len(t) < 2:
            return np.tile(x0, (len(t), 1))
        rel_t = t - t[0]
        sim_kwargs = {}
        if u is not None:
            control = np.asarray(u, dtype=float)
            if control.ndim == 1:
                control = control[:, None]
            sim_kwargs["u"] = control
        try:
            traj = self._model.simulate(x0, rel_t, **sim_kwargs)
        except Exception as err:
            raise RuntimeError("PySINDy の simulate 実行に失敗しました") from err
        return np.asarray(traj, dtype=float)


@register_model
class PySINDyModel(_PySINDyAdapter):
    """PySINDy の標準 SINDy/STLSQ を利用するモデル"""

    name = "pysindy"
    mode = "sindy"


@register_model
class PySINDyPIModel(_PySINDyAdapter):
    """PySINDy の SINDy-PI 版。粗いサンプリング向けの既定候補"""

    name = "pysindy_pi"
    mode = "sindy_pi"
