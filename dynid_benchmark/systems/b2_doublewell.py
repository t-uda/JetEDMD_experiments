import numpy as np
from .base import DynamicalSystem, euler_maruyama


class DoubleWellSDE(DynamicalSystem):
    def simulate_true(self, T: float, dt_true: float, seed=None):
        p = self.params
        sigma = p.get("sigma", 0.3)
        rng = np.random.default_rng(seed)
        x0 = np.array([rng.normal()])  # 二重井戸ポテンシャル周りの初期値

        def g(t, x):
            drift = -(x**3 - x)  # 勾配流（ダブルウェルのポテンシャル）
            return drift, np.array([sigma])

        return euler_maruyama(g, x0, T, dt_true, rng)
