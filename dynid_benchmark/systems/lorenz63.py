import numpy as np

from .base import DynamicalSystem, simulate_ode


class Lorenz63(DynamicalSystem):
    """古典的なローレンツ63 系（パラメータを外部から調整可能）"""

    def simulate_true(self, T: float, dt_true: float, seed=None):
        p = self.params
        sigma = p.get("sigma", 10.0)
        rho = p.get("rho", 28.0)
        beta = p.get("beta", 8.0 / 3.0)

        def f(_, state):
            x, y, z = state
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return np.array([dx, dy, dz])

        rng = np.random.default_rng(seed)
        x0 = np.array(
            [
                1.0 + 0.1 * rng.standard_normal(),
                1.0 + 0.1 * rng.standard_normal(),
                1.0 + 0.1 * rng.standard_normal(),
            ]
        )  # アトラクタ近傍にランダムに初期化
        return simulate_ode(f, x0, T, dt_true)
