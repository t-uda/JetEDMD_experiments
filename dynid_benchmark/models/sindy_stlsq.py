import numpy as np
from .base import Model, register_model


def finite_difference(y, t):
    dy = np.zeros_like(y)
    dt = np.diff(t)
    dt = np.clip(dt, 1e-12, None)
    # 内点は中心差分、端点は前進／後退差分で近似微分を計算
    for i in range(y.shape[1]):
        yi = y[:, i]
        dyi = dy[:, i]
        dyi[1:-1] = (yi[2:] - yi[:-2]) / (t[2:] - t[:-2])
        dyi[0] = (yi[1] - yi[0]) / dt[0]
        dyi[-1] = (yi[-1] - yi[-2]) / dt[-1]
    return dy


def build_library(y, poly_order=3, include_sin_cos=False):
    N, d = y.shape
    Theta = [np.ones((N, 1))]
    names = ["1"]
    # 指定次数までの多項式基底を全て列挙
    for p in range(1, poly_order + 1):
        # 次数 p の全単項式を生成
        from itertools import combinations_with_replacement

        for combo in combinations_with_replacement(range(d), p):
            term = np.prod([y[:, j] for j in combo], axis=0)[:, None]
            Theta.append(term)
            name = "*".join([f"x{j}" for j in combo])
            names.append(name)
    if include_sin_cos:
        for j in range(d):
            Theta.append(np.sin(y[:, j : j + 1]))
            names.append(f"sin(x{j})")
            Theta.append(np.cos(y[:, j : j + 1]))
            names.append(f"cos(x{j})")
    Theta = np.concatenate(Theta, axis=1)
    return Theta, names


def stlsq(Theta, dXdt, lam=0.1, max_iter=10):
    # 逐次しきい値処理付き最小二乗法で疎な係数行列 Xi を推定
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(Xi) < lam
        Xi[small] = 0.0
        for k in range(dXdt.shape[1]):
            big_idx = ~small[:, k]
            if np.sum(big_idx) == 0:
                continue
            Xi[big_idx, k] = np.linalg.lstsq(Theta[:, big_idx], dXdt[:, k], rcond=None)[
                0
            ]
    return Xi


@register_model
class SINDySTLSQ(Model):
    name = "sindy_stlsq"

    def __init__(
        self, poly_order=3, include_sin_cos=False, lam=0.1, max_iter=10, smooth_window=0
    ):
        super().__init__(
            poly_order=poly_order,
            include_sin_cos=include_sin_cos,
            lam=lam,
            max_iter=max_iter,
            smooth_window=smooth_window,
        )
        self.poly_order = poly_order
        self.include_sin_cos = include_sin_cos
        self.lam = lam
        self.max_iter = max_iter
        self.smooth_window = smooth_window
        self.Xi = None
        self.Theta_names = None

    def _smooth(self, y, win):
        if win <= 1:
            return y
        from numpy.lib.stride_tricks import sliding_window_view

        if win % 2 == 0:
            win += 1
        pad = win // 2
        ypad = np.pad(y, ((pad, pad), (0, 0)), mode="edge")
        sw = sliding_window_view(ypad, (win, y.shape[1]))[:, 0, :, :]
        return np.mean(sw, axis=1)  # エッジを複製した移動平均フィルタ

    def fit(self, t, y, u=None):
        y_use = self._smooth(y, self.smooth_window)
        # 平滑化した系列から有限差分とライブラリ行列を構築
        dXdt = finite_difference(y_use, t)
        Theta, names = build_library(y_use, self.poly_order, self.include_sin_cos)
        Xi = stlsq(Theta, dXdt, lam=self.lam, max_iter=self.max_iter)
        self.Xi = Xi
        self.Theta_names = names

    def _eval_theta(self, x_row):
        x = x_row[None, :]
        Theta_row, _ = build_library(x, self.poly_order, self.include_sin_cos)
        return Theta_row[0]

    def predict_derivative(self, t, x, u=None):
        theta = self._eval_theta(x)
        return theta @ (self.Xi)
