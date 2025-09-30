from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Callable
import numpy as np


class DynamicalSystem(ABC):
    """真値軌道の生成と観測サンプリングの基底インターフェース"""

    def __init__(self, params: Dict):
        self.params = params

    @abstractmethod
    def simulate_true(
        self, T: float, dt_true: float, seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """戻り値の辞書には以下が含まれる:
        - 't': 時刻列 (shape [N])
        - 'x': 状態軌道 (shape [N, d])
        - 入力を持つ場合は 'u': 入力系列 (shape [N, m])
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
        """サブサンプリング・ジッタ・雑音加算を行い観測データを生成"""
        rng = np.random.default_rng(seed)
        t_true = true["t"]
        x_true = true["x"]
        u_true = true.get("u", None)
        dt_obs = r * Tfast

        # 真値系列の時間範囲全体で等間隔の観測グリッドを構築
        t0, t1 = float(t_true[0]), float(t_true[-1])
        n_obs = int(np.floor((t1 - t0) / dt_obs)) + 1
        t_obs = t0 + np.arange(n_obs) * dt_obs

        # 観測グリッドに揺らぎを加える（割合は dt_obs に対する jitter_pct）
        if jitter_pct > 0:
            jitter = (rng.random(size=t_obs.shape) * 2 - 1) * (jitter_pct * dt_obs)
            t_obs = np.clip(t_obs + jitter, t0, t1)

        # 観測時刻に合わせて真値の状態・入力を線形補間する補助関数
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

        # 次元ごとの分散から所望の SNR を満たすガウス雑音を加算
        if snr_db is not None:
            var = np.var(y_obs, axis=0) + 1e-12
            sigma = np.sqrt(var / (10 ** (snr_db / 10.0)))
            noise = rng.normal(size=y_obs.shape) * sigma
            y_obs = y_obs + noise

        # 外れ値を指定割合だけ強調する（観測行を 10 倍に置換）
        if outlier_rate and outlier_rate > 0:
            M = y_obs.shape[0]
            k = int(M * outlier_rate)
            if k > 0:
                idx = rng.choice(M, size=k, replace=False)
                y_obs[idx] = y_obs[idx] * 10.0

        # 欠測を指定割合で導入（該当する時刻と観測値を除去）
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
        # 評価時の参照に使えるよう真値系列も保持しておく
        out["_true_t"] = t_true
        out["_true_x"] = x_true
        if u_true is not None:
            out["_true_u"] = u_true
        return out


# Generic integrators (RK4 and Euler-Maruyama)
def rk4_step(f, t, x, dt):
    # 4 次のルンゲ=クッタ法を 1 ステップ進める共通ユーティリティ
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
            ui = u_fn(ti, x[i - 1])  # 入力付き系では同じ時刻の入力を取得

            def f_pack(tj, xj):
                return f(tj, xj, ui)  # 入力を固定したベクトル場に包む

            x[i] = rk4_step(f_pack, ti, x[i - 1], dt)
            u[i] = ui
        else:
            x[i] = rk4_step(f, ti, x[i - 1], dt)  # 入力なしの純粋な系を積分
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
        drift, sigma = g(ti, xi)  # 漂移項と拡散係数（スカラーまたは次元ごと）を取得
        dW = rng.normal(size=xi.shape) * np.sqrt(dt)
        x[i] = xi + drift * dt + sigma * dW
    return {"t": t, "x": x}
