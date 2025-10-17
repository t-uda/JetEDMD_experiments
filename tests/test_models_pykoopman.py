import numpy as np
import pytest

pytest.importorskip("pykoopman", reason="PyKoopman dependency not installed")

from dynid_benchmark.models.pykoopman_adapter import (
    PyKoopmanEDMD,
    PyKoopmanEDMDc,
)

from .model_test_utils import discrete_trajectory, relative_l2


pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The attribute `n_input_features_` was deprecated:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Casting complex values to real discards the imaginary part:numpy.ComplexWarning"
    ),
]


def test_pykoopman_edmd_recovers_linear_map():
    t, y = discrete_trajectory(factors=(0.9,), dt=0.1, n=80)
    model = PyKoopmanEDMD(poly_order=1)
    model.fit(t, y)
    y_hat = model.rollout(t, y[0])
    assert np.all(np.isfinite(y_hat))
    assert relative_l2(y_hat, y) < 5e-3


def test_pykoopman_edmdc_tracks_controlled_linear_system():
    dt = 0.05
    n = 60
    t = np.arange(n, dtype=float) * dt
    u = np.sin(t)[:, None]
    x = np.zeros((n, 1), dtype=float)
    A = 0.7
    B = 0.4
    for k in range(n - 1):
        x[k + 1, 0] = A * x[k, 0] + B * u[k, 0]

    model = PyKoopmanEDMDc(poly_order=1)
    model.fit(t, x, u)
    y_hat = model.rollout(t, x[0], u)
    assert np.all(np.isfinite(y_hat))
    assert relative_l2(y_hat, x) < 1e-2


def test_pykoopman_requires_uniform_timestep():
    t = np.array([0.0, 0.1, 0.25])
    y = np.zeros((3, 1))
    model = PyKoopmanEDMD()
    with pytest.raises(ValueError):
        model.fit(t, y)
