from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Callable
import numpy as np


class DynamicalSystem(ABC):
    """Base interface for generating ground-truth trajectories and sampling."""

    def __init__(self, params: Dict):
        self.params = params

    @abstractmethod
    def simulate_true(
        self, T: float, dt_true: float, seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Return dict with keys:
        - 't': time array (shape [N])
        - 'x': state array (shape [N, d])
        - optionally 'u': input array (shape [N, m]) if system has input
        """
        pass

    def sample_observations(
        self,
        true: Dict[str, np.ndarray],
        Tfast: float,
        r: float,
        snr_db: float = 30.0,
        jitter_pct: float = 0.0,
        missing_pct: float = 0.0,
        outlier_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Subsample, jitter, add noise/outliers. Returns dict with 't','y', optionally 'u'."""
        rng = np.random.default_rng(seed)
        t_true = true["t"]
        x_true = true["x"]
        u_true = true.get("u", None)
        dt_obs = r * Tfast

        # Build uniform grid on [t_true[0], t_true[-1]]
        t0, t1 = float(t_true[0]), float(t_true[-1])
        n_obs = int(np.floor((t1 - t0) / dt_obs)) + 1
        t_obs = t0 + np.arange(n_obs) * dt_obs

        # Optional jitter (± jitter_pct of dt_obs)
        if jitter_pct > 0:
            jitter = (rng.random(size=t_obs.shape) * 2 - 1) * (jitter_pct * dt_obs)
            t_obs = np.clip(t_obs + jitter, t0, t1)

        # Interpolate states (linear) to observation times
        def interp_traj(ts, xs, tq):
            # xs: [N, d], ts: [N], tq: [M]
            d = xs.shape[1]
            y = np.empty((len(tq), d), dtype=float)
            idx = np.searchsorted(ts, tq, side="left")
            idx = np.clip(idx, 1, len(ts) - 1)
            t0s = ts[idx - 1]
            t1s = ts[idx]
            w = (tq - t0s) / (t1s - t0s + 1e-12)
            y = (1.0 - w)[:, None] * xs[idx - 1] + w[:, None] * xs[idx]
            return y

        y_obs = interp_traj(t_true, x_true, t_obs)
        if u_true is not None:
            u_obs = interp_traj(t_true, u_true, t_obs)
        else:
            u_obs = None

        # Additive Gaussian noise to achieve SNR (per-dimension)
        if snr_db is not None:
            var = np.var(y_obs, axis=0) + 1e-12
            sigma = np.sqrt(var / (10 ** (snr_db / 10.0)))
            noise = rng.normal(size=y_obs.shape) * sigma
            y_obs = y_obs + noise

        # Outliers: replace random rows with amplified values
        if outlier_rate and outlier_rate > 0:
            M = y_obs.shape[0]
            k = int(M * outlier_rate)
            if k > 0:
                idx = rng.choice(M, size=k, replace=False)
                y_obs[idx] = y_obs[idx] * 10.0

        # Missingness: drop random rows
        if missing_pct and missing_pct > 0:
            M = len(t_obs)
            k = int(M * missing_pct)
            if k > 0:
                idx = np.sort(rng.choice(M, size=k, replace=False))
                mask = np.ones(M, dtype=bool)
                mask[idx] = False
                t_obs = t_obs[mask]
                y_obs = y_obs[mask]
                if u_obs is not None:
                    u_obs = u_obs[mask]

        out = {"t": t_obs, "y": y_obs}
        if u_obs is not None:
            out["u"] = u_obs
        # Save reference (optional) for evaluation convenience
        out["_true_t"] = t_true
        out["_true_x"] = x_true
        if u_true is not None:
            out["_true_u"] = u_true
        return out


# Generic integrators (RK4 and Euler-Maruyama)
def rk4_step(f, t, x, dt):
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_ode(f, x0, T, dt, with_u: bool = False, u_fn=None):
    import numpy as np

    N = int(np.floor(T / dt)) + 1
    t = np.linspace(0.0, N * dt, N)
    x = np.zeros((N, len(x0)), dtype=float)
    x[0] = x0
    if with_u and u_fn is not None:
        u = np.zeros((N, u_fn(0.0, x0).shape[0]), dtype=float)
        u[0] = u_fn(0.0, x0)
    else:
        u = None
    for i in range(1, N):
        ti = t[i - 1]
        if with_u and u_fn is not None:
            ui = u_fn(ti, x[i - 1])

            def f_pack(tj, xj):
                return f(tj, xj, ui)

            x[i] = rk4_step(f_pack, ti, x[i - 1], dt)
            u[i] = ui
        else:
            x[i] = rk4_step(f, ti, x[i - 1], dt)
    if u is not None:
        return {"t": t, "x": x, "u": u}
    else:
        return {"t": t, "x": x}


def euler_maruyama(g, x0, T, dt, rng):
    import numpy as np

    N = int(np.floor(T / dt)) + 1
    t = np.linspace(0.0, N * dt, N)
    x = np.zeros((N, len(x0)), dtype=float)
    x[0] = x0
    for i in range(1, N):
        xi = x[i - 1]
        ti = t[i - 1]
        drift, sigma = g(ti, xi)  # returns (drift vector, scalar sigma or per-dim)
        dW = rng.normal(size=xi.shape) * np.sqrt(dt)
        x[i] = xi + drift * dt + sigma * dW
    return {"t": t, "x": x}
