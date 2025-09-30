import numpy as np
import pytest

from dynid_benchmark.systems.lorenz63 import Lorenz63
from dynid_benchmark.systems.c1_lti_mass_spring import MassSpringInput
from dynid_benchmark.systems.d1_burgers import Burgers1D
from dynid_benchmark.systems.a1_dry_friction import DryFrictionOscillator


def relative_l2(a, b):
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return np.linalg.norm(a)
    return np.linalg.norm(a - b) / denom


@pytest.mark.parametrize("seed", [0, 42])
def test_lorenz63_produces_finite_trajectory(seed):
    sys = Lorenz63({})
    out = sys.simulate_true(T=0.2, dt_true=1e-3, seed=seed)
    assert out["x"].shape[1] == 3
    assert np.all(np.isfinite(out["x"]))
    assert out["x"].shape[0] > 0


@pytest.mark.parametrize("mode", ["prbs", "sine", "sweep", "chirp"])
def test_mass_spring_input_modes(mode):
    sys = MassSpringInput({"input_mode": mode, "zeta": 0.02, "omega": 2 * np.pi})
    out = sys.simulate_true(T=0.5, dt_true=1e-3, seed=101)
    assert out["x"].shape[1] == 2
    assert out.get("u") is not None
    assert np.all(np.isfinite(out["x"]))


def test_burgers1d_observation_shapes():
    sys = Burgers1D({"nu": 0.00318, "L": 1.0, "Nx": 64})
    true = sys.simulate_true(T=0.05, dt_true=2e-4)
    obs = sys.sample_observations(true, Tfast=0.1, r=0.1, snr_db=30.0, seed=123)
    assert obs["y"].ndim == 2
    assert obs["y"].shape[0] > 0
    assert np.all(np.isfinite(obs["y"]))


@pytest.mark.parametrize("kappa", [5.0, 20.0])
def test_dry_friction_sampler_handles_kappa(kappa):
    sys = DryFrictionOscillator({"k": 39.48, "kappa": kappa})
    true = sys.simulate_true(T=0.2, dt_true=1e-3, seed=202)
    obs = sys.sample_observations(true, Tfast=1.0, r=0.1, snr_db=30.0, seed=202)
    assert obs["y"].shape[1] == 2
    assert np.all(np.isfinite(obs["y"]))
