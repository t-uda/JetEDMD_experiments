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


_SINDY_EXPLICIT_CASES = [
    pytest.param(
        "sindy_stlsq",
        dict(poly_order=1, include_sin_cos=False, lam=1e-3, max_iter=6),
        2e-2,
        id="sindy_stlsq_p1",
    ),
    pytest.param(
        "sindy_stlsq",
        dict(poly_order=2, include_sin_cos=False, lam=5e-3, max_iter=6),
        2e-2,
        id="sindy_stlsq_p2",
    ),
]

if _PYSINDY_OK:
    _SINDY_EXPLICIT_CASES.append(
        pytest.param(
            "pysindy",
            dict(poly_order=1, optimizer_kwargs={"threshold": 1e-3}),
            5e-2,
            id="pysindy",
        )
    )
else:  # pragma: no cover - dependency missing path
    _SINDY_EXPLICIT_CASES.append(
        pytest.param(
            "pysindy",
            {},
            5e-2,
            id="pysindy",
            marks=pytest.mark.skip(reason="PySINDy dependency not installed"),
        )
    )


@pytest.mark.parametrize("model_name, kwargs, tol", _SINDY_EXPLICIT_CASES)
def test_sindy_explicit_variants_recovers_linear_decay(model_name, kwargs, tol):
    t, y = linear_trajectory(dims=1)
    if model_name == "sindy_stlsq":
        model = SINDySTLSQ(**kwargs)
    elif model_name == "pysindy":
        model = pysindy_mod.PySINDyModel(**kwargs)  # type: ignore[attr-defined]
    else:  # pragma: no cover - safety guard for new variants
        pytest.fail(f"unsupported model_name={model_name}")
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < tol


_SINDY_PI_CASES = [
    pytest.param(
        "sindy_pi",
        dict(poly_order=1, window_len=3, lam=1e-3, max_iter=6, ridge=1e-10),
        1e-2,
        id="sindy_pi_w3",
    ),
    pytest.param(
        "sindy_pi",
        dict(poly_order=1, window_len=5, lam=1e-3, max_iter=6, ridge=1e-10),
        1e-2,
        id="sindy_pi_w5",
    ),
]

if _PYSINDY_OK:
    marks = [
        pytest.mark.skipif(
            not _PYSINDY_PI_OK,
            reason="PySINDy-PI (cvxpy) dependency not installed",
        ),
        pytest.mark.xfail(
            reason="PySINDy-PI (1.x) simulateが上流ライブラリで不安定", strict=False
        ),
    ]
    _SINDY_PI_CASES.append(
        pytest.param(
            "pysindy_pi",
            dict(poly_order=1, optimizer_kwargs={"max_iter": 10}),
            5e-2,
            id="pysindy_pi",
            marks=marks,
        )
    )
else:  # pragma: no cover - dependency missing path
    _SINDY_PI_CASES.append(
        pytest.param(
            "pysindy_pi",
            {},
            5e-2,
            id="pysindy_pi",
            marks=pytest.mark.skip(reason="PySINDy dependency not installed"),
        )
    )


@pytest.mark.parametrize("model_name, kwargs, tol", _SINDY_PI_CASES)
def test_sindy_pi_variants_recovers_linear_decay(model_name, kwargs, tol):
    t, y = linear_trajectory(dims=1)
    if model_name == "sindy_pi":
        model = SINDyPI(**kwargs)
    elif model_name == "pysindy_pi":
        model = pysindy_mod.PySINDyPIModel(**kwargs)  # type: ignore[attr-defined]
    else:  # pragma: no cover - safety guard for new variants
        pytest.fail(f"unsupported model_name={model_name}")
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < tol


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

