import numpy as np
from .base import DynamicalSystem, simulate_ode

class OrnsteinUhlenbeck(DynamicalSystem):
    def simulate_true(self, T: float, dt_true: float, seed=None):
        p = self.params
        theta = p.get("theta", 1.0)

        def f(t, x):
            return -theta * x

        rng = np.random.default_rng(seed)
        x0 = np.array([rng.normal()])
        return simulate_ode(lambda t, x: f(t, x), x0, T, dt_true)
