import numpy as np
from .base import DynamicalSystem, simulate_ode


class DuffingForced(DynamicalSystem):
    def simulate_true(self, T: float, dt_true: float, seed=None):
        p = self.params
        delta = p.get("delta", 0.2)
        alpha = p.get("alpha", -1.0)
        beta = p.get("beta", 1.0)
        gamma = p.get("gamma", 0.3)
        Omega = p.get("Omega", 1.2)

        def f(t, s):
            x, v = s
            a = -delta * v - alpha * x - beta * (x**3) + gamma * np.cos(Omega * t)
            return np.array([v, a])

        x0 = np.array([0.0, 0.0])
        return simulate_ode(f, x0, T, dt_true)
