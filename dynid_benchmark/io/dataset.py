import numpy as np
from typing import Dict, Tuple
from ..systems.base import DynamicalSystem


def split_traj(obs: Dict[str, np.ndarray], train: float, val: float, test: float):
    N = len(obs["t"])
    i1 = int(N * train)
    i2 = int(N * (train + val))
    out = {}
    out["train"] = {
        k: (v[:i1] if isinstance(v, np.ndarray) and len(v) == N else v)
        for k, v in obs.items()
        if not k.startswith("_")
    }
    out["val"] = {
        k: (v[i1:i2] if isinstance(v, np.ndarray) and len(v) == N else v)
        for k, v in obs.items()
        if not k.startswith("_")
    }
    out["test"] = {
        k: (v[i2:] if isinstance(v, np.ndarray) and len(v) == N else v)
        for k, v in obs.items()
        if not k.startswith("_")
    }
    return out


def to_snr_db(y, snr_db, rng):
    var = np.var(y, axis=0) + 1e-12
    sigma = np.sqrt(var / (10 ** (snr_db / 10.0)))
    return y + rng.normal(size=y.shape) * sigma
