import numpy as np
import pytest

from dynid_benchmark.models.sindy_stlsq import SINDySTLSQ
from dynid_benchmark.models.sindy_pi import SINDyPI
from dynid_benchmark.models.sindy_implicit import ImplicitSINDy
from dynid_benchmark.models.edmd import EDMD

try:  # pragma: no cover - optional dependency guard
    from dynid_benchmark.models import pysindy_adapter as pysindy_mod

    _PYSINDY_OK = pysindy_mod._PYSINDY_OK
    _PYSINDY_PI_OK = getattr(pysindy_mod, "_CVXPY_OK", False)
except Exception:  # pragma: no cover - fallback when pysindy missing
    pysindy_mod = None  # type: ignore
    _PYSINDY_OK = False
    _PYSINDY_PI_OK = False


def linear_trajectory(decay=-1.0, t_end=2.0, n=201, dims=1):
    t = np.linspace(0.0, t_end, n)
    base = np.exp(decay * t)
    y = np.column_stack([base ** (k + 1) for k in range(dims)])
    return t, y


def discrete_trajectory(factors=(0.9,), dt=0.1, n=120):
    t = np.arange(n, dtype=float) * dt
    powers = np.arange(n)
    y = np.column_stack([f**powers for f in factors])
    return t, y


def relative_l2(a, b):
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return np.linalg.norm(a)
    return np.linalg.norm(a - b) / denom


@pytest.mark.parametrize(
    "poly_order, lam, include_sin_cos",
    [
        (1, 1e-3, False),
        (2, 5e-3, False),
    ],
)
def test_sindy_stlsq_recovers_linear_decay(poly_order, lam, include_sin_cos):
    t, y = linear_trajectory(dims=1)
    model = SINDySTLSQ(
        poly_order=poly_order, include_sin_cos=include_sin_cos, lam=lam, max_iter=6
    )
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 2e-2


@pytest.mark.parametrize("window_len", [3, 5])
def test_sindy_pi_recovers_linear_decay(window_len):
    t, y = linear_trajectory(dims=1)
    model = SINDyPI(
        poly_order=1, window_len=window_len, lam=1e-3, max_iter=6, ridge=1e-10
    )
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 1e-2


@pytest.mark.parametrize("denom_order", [0, 1])
def test_sindy_implicit_recovers_linear_decay(denom_order):
    t, y = linear_trajectory(dims=1)
    model = ImplicitSINDy(
        poly_order=1, denom_order=denom_order, include_sin_cos=False, thresh=1e-6
    )
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 2e-2


@pytest.mark.parametrize("factors, order", [((0.9,), 1), ((0.9, 0.8), 2)])
def test_edmd_linear_map_rollout_matches_truth(factors, order):
    t, y = discrete_trajectory(factors=factors)
    model = EDMD(order=order, ridge=1e-10)
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 2e-2


def test_sindy_pi_requires_sufficient_window_samples():
    t, y = linear_trajectory(t_end=0.05, n=5)
    model = SINDyPI(window_len=5)
    with pytest.raises(ValueError):
        model.fit(t, y)


@pytest.mark.skipif(not _PYSINDY_OK, reason="PySINDy dependency not installed")
def test_pysindy_adapter_matches_linear_rollout():
    t, y = linear_trajectory(dims=1)
    model = pysindy_mod.PySINDyModel(poly_order=1, optimizer_kwargs={"threshold": 1e-3})
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 5e-2


@pytest.mark.skipif(
    not (_PYSINDY_OK and _PYSINDY_PI_OK),
    reason="PySINDy-PI (cvxpy) dependency not installed",
)
@pytest.mark.xfail(reason="PySINDy-PI (1.x) simulateが上流ライブラリで不安定", strict=False)
def test_pysindy_pi_adapter_runs_without_nan():
    t, y = linear_trajectory(dims=1)
    model = pysindy_mod.PySINDyPIModel(poly_order=1, optimizer_kwargs={"max_iter": 10})
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    assert np.all(np.isfinite(y_hat))
