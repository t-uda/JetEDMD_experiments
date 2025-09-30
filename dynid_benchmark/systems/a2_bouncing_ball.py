import numpy as np
from .base import DynamicalSystem


class BouncingBall(DynamicalSystem):
    """パラメータ mode に応じて硬い衝突／ソフトコンタクトを切り替えるモデル"""

    def simulate_true(self, T: float, dt_true: float, seed=None):
        p = self.params
        mode = p.get("mode", "hard")
        g = p.get("g", 9.81)
        r = p.get("restitution", 0.85)
        m = p.get("m", 1.0)
        ks = p.get("k_s", 1e3)
        zeta = p.get("zeta_s", 0.05)
        cs = 2 * zeta * np.sqrt(ks * m)

        y0 = p.get("y0", 1.0)
        v0 = p.get("v0", 0.0)
        N = int(np.floor(T / dt_true)) + 1
        t = np.linspace(0.0, N * dt_true, N)
        y = np.zeros(N)
        v = np.zeros(N)
        y[0] = y0
        v[0] = v0

        for i in range(1, N):
            if mode == "hard":
                # シンプレクティック・オイラーで重力落下を更新し接触イベントを処理
                v[i] = v[i - 1] - g * dt_true
                y[i] = y[i - 1] + v[i] * dt_true
                if y[i] < 0.0 and v[i] < 0.0:
                    # 接触時刻を線形に補間し、反発係数を適用
                    alpha = y[i - 1] / (y[i - 1] - y[i] + 1e-12)
                    y[i] = 0.0
                    v_contact = v[i - 1] - g * (alpha * dt_true)
                    v[i] = -r * v_contact
            else:
                # y<0 の間はバネ・ダンパで地面とのソフトコンタクトを表現
                a = -g
                if y[i - 1] < 0.0:
                    a += -(ks / m) * (y[i - 1]) - (cs / m) * v[i - 1]
                v[i] = v[i - 1] + a * dt_true
                y[i] = y[i - 1] + v[i] * dt_true
        x = np.stack([y, v], axis=1)
        return {"t": t, "x": x}
