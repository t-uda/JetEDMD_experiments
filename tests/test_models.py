import numpy as np

from dynid_benchmark.models.sindy_stlsq import SINDySTLSQ
from dynid_benchmark.models.sindy_pi import SINDyPI
from dynid_benchmark.models.sindy_implicit import ImplicitSINDy
from dynid_benchmark.models.edmd import EDMD


def linear_trajectory(decay=-1.0, t_end=2.0, n=201):
    t = np.linspace(0.0, t_end, n)
    x = np.exp(decay * t)
    return t, x[:, None]


def discrete_trajectory(factor=0.9, dt=0.1, n=120):
    t = np.arange(n, dtype=float) * dt
    x = factor ** np.arange(n)
    return t, x[:, None]


def relative_l2(a, b):
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return np.linalg.norm(a)
    return np.linalg.norm(a - b) / denom


def test_sindy_stlsq_recovers_linear_decay():
    t, y = linear_trajectory()
    model = SINDySTLSQ(poly_order=1, lam=1e-3, max_iter=5)
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 1e-2


def test_sindy_pi_recovers_linear_decay():
    t, y = linear_trajectory()
    model = SINDyPI(poly_order=1, window_len=4, lam=1e-3, max_iter=5, ridge=1e-10)
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 5e-3


def test_sindy_implicit_recovers_linear_decay():
    t, y = linear_trajectory()
    model = ImplicitSINDy(poly_order=1, denom_order=0, thresh=1e-6)
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 1e-2


def test_edmd_linear_map_rollout_matches_truth():
    t, y = discrete_trajectory()
    model = EDMD(order=1, ridge=1e-10)
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    err = relative_l2(y_hat, y)
    assert np.all(np.isfinite(y_hat))
    assert err < 5e-3
