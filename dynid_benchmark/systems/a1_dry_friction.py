import numpy as np
from .base import DynamicalSystem, simulate_ode


class DryFrictionOscillator(DynamicalSystem):
    def simulate_true(self, T: float, dt_true: float, seed=None):
        p = self.params
        m = p.get("m", 1.0)
        k = p["k"]
        zeta = p.get("zeta", 0.02)
        c = 2.0 * zeta * np.sqrt(k * m)
        mu = p.get("mu", 0.2)
        g = p.get("g", 9.81)
        kappa = p.get("kappa", 20.0)  # smoothing of sgn

        def f(t, s):
            x, v = s
            f_fric = mu * g * np.tanh(kappa * v)  # smooth sgn
            a = -(c / m) * v - (k / m) * x - f_fric
            return np.array([v, a])

        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0.5, 1.0)
        v0 = 0.0
        return simulate_ode(f, np.array([x0, v0]), T, dt_true)
