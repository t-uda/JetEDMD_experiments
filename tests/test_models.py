import numpy as np
import pytest

from dynid_benchmark.models.sindy_stlsq import SINDySTLSQ
from dynid_benchmark.models.sindy_pi import SINDyPI
from dynid_benchmark.models.sindy_implicit import ImplicitSINDy
from dynid_benchmark.models.edmd import EDMD

from .model_test_utils import (
    discrete_trajectory,
    linear_trajectory,
    relative_l2,
)


@pytest.mark.parametrize(
    "kwargs,tol",
    [
        pytest.param(
            dict(poly_order=1, include_sin_cos=False, lam=1e-3, max_iter=6),
            2e-2,
            id="sindy_stlsq_p1",
        ),
        pytest.param(
            dict(poly_order=2, include_sin_cos=False, lam=5e-3, max_iter=6),
            2e-2,
            id="sindy_stlsq_p2",
        ),
    ],
)
def test_sindy_stlsq_recovers_linear_decay(kwargs, tol):
    t, y = linear_trajectory(dims=1)
    model = SINDySTLSQ(**kwargs)
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < tol


@pytest.mark.parametrize(
    "kwargs,tol",
    [
        pytest.param(
            dict(poly_order=1, window_len=3, lam=1e-3, max_iter=6, ridge=1e-10),
            1e-2,
            id="sindy_pi_w3",
        ),
        pytest.param(
            dict(poly_order=1, window_len=5, lam=1e-3, max_iter=6, ridge=1e-10),
            1e-2,
            id="sindy_pi_w5",
        ),
    ],
)
def test_sindy_pi_recovers_linear_decay(kwargs, tol):
    t, y = linear_trajectory(dims=1)
    model = SINDyPI(**kwargs)
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

