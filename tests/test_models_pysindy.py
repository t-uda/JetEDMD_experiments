import numpy as np
import pytest

pytest.importorskip("pysindy", reason="PySINDy dependency not installed")

from dynid_benchmark.models import pysindy_adapter as pysindy_mod

from .model_test_utils import linear_trajectory, relative_l2


@pytest.mark.parametrize(
    "kwargs,tol",
    [
        pytest.param(
            dict(poly_order=1, optimizer_kwargs={"threshold": 1e-3}),
            5e-2,
            id="pysindy",
        ),
    ],
)
def test_pysindy_recovers_linear_decay(kwargs, tol):
    t, y = linear_trajectory(dims=1)
    model = pysindy_mod.PySINDyModel(**kwargs)
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < tol


@pytest.mark.skipif(
    not getattr(pysindy_mod, "_CVXPY_OK", False),
    reason="PySINDy-PI (cvxpy) dependency not installed",
)
@pytest.mark.xfail(reason="PySINDy-PI simulate が不安定", strict=False)
def test_pysindy_pi_recovers_linear_decay():
    t, y = linear_trajectory(dims=1)
    model = pysindy_mod.PySINDyPIModel(poly_order=1, optimizer_kwargs={"max_iter": 10})
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 5e-2
