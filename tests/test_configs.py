import numpy as np
import pytest

from dynid_benchmark.config import load_yaml
from dynid_benchmark.runners.run_experiment import SYSTEMS


@pytest.mark.parametrize(
    "config_path",
    [
        "exp/E1_Lorenz63.yaml",
        "exp/E2_MSD_Input.yaml",
        "exp/E3_Burgers1D.yaml",
        "exp/E4_DryFriction.yaml",
    ],
)
def test_core4_configs_resolve_system(config_path):
    cfg = load_yaml(config_path)
    assert cfg.system in SYSTEMS
    assert cfg.r_list
    assert cfg.noise["SNR_dB"]


def test_lorenz63_config_has_metrics():
    cfg = load_yaml("exp/E1_Lorenz63.yaml")
    assert "rollout_rmse" in cfg.metrics
