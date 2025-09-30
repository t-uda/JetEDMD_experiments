import numpy as np
from .base import Model, register_model

def finite_difference(y, t):
    """Compute numerical time-derivative with simple central differences.
    Edges use one-sided differences. Returns array with same shape as y."""
    dy = np.zeros_like(y, dtype=float)
    dt = np.diff(t)
    dt = np.clip(dt, 1e-12, None)
    # central for interior, forward/backward for edges
    dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])[:,None]
    dy[0]     = (y[1]  - y[0])   / dt[0]
    dy[-1]    = (y[-1] - y[-2])  / dt[-1]
    return dy

def build_library(y, poly_order=3, include_sin_cos=False):
    """Polynomial (+ optional sin/cos) library with constant term."""
    N, d = y.shape
    Theta = [np.ones((N,1))]
    from itertools import combinations_with_replacement
    for p in range(1, poly_order+1):
        for combo in combinations_with_replacement(range(d), p):
            Theta.append(np.prod([y[:,j] for j in combo], axis=0)[:,None])
    if include_sin_cos:
        for j in range(d):
            Theta.append(np.sin(y[:,j:j+1]))
            Theta.append(np.cos(y[:,j:j+1]))
    return np.concatenate(Theta, axis=1)

def scale_columns(A, eps=1e-12):
    """Column L2 scaling to unit norm for conditioning."""
    s = np.linalg.norm(A, axis=0)
    s = np.where(s < eps, 1.0, s)
    return A / s, s

def stlsq(Theta, dXdt, lam=0.05, max_iter=10, ridge=1e-8):
    """Sequentially thresholded least squares with small ridge regularization.
    - ridge: avoid singular normal equations
    - lam:   threshold value for sparsity
    """
    # Initial ridge solution
    G = Theta.T @ Theta + ridge*np.eye(Theta.shape[1])
    Xi = np.linalg.solve(G, Theta.T @ dXdt)
    for _ in range(max_iter):
        small = np.abs(Xi) < lam
        Xi[small] = 0.0
        for k in range(dXdt.shape[1]):
            big = ~small[:,k]
            if np.sum(big) == 0: 
                continue
            # Refit active set with ridge
            Gk = Theta[:,big].T @ Theta[:,big] + ridge*np.eye(np.sum(big))
            Xi[big, k] = np.linalg.solve(Gk, Theta[:,big].T @ dXdt[:,k])
    return Xi

@register_model
class SINDySTLSQ(Model):
    """Classic SINDy with STLSQ; stabilized by column scaling and ridge."""
    name = "sindy_stlsq"

    def __init__(self, poly_order=3, include_sin_cos=False, lam=0.05, max_iter=10, smooth_window=0, ridge=1e-8):
        super().__init__(poly_order=poly_order, include_sin_cos=include_sin_cos,
                         lam=lam, max_iter=max_iter, smooth_window=smooth_window, ridge=ridge)
        self.poly_order = poly_order
        self.include_sin_cos = include_sin_cos
        self.lam = lam
        self.max_iter = max_iter
        self.smooth_window = smooth_window
        self.ridge = ridge
        self.Xi = None
        self.scale = None  # feature scaling used in training

    def _smooth(self, y, win):
        """Simple moving-average smoothing (optional) to mitigate noise amplification in differencing."""
        if win <= 1: 
            return y
        if win % 2 == 0: 
            win += 1
        pad = win//2
        ypad = np.pad(y, ((pad,pad),(0,0)), mode="edge")
        # Sliding mean
        sw = np.lib.stride_tricks.sliding_window_view(ypad, (win, y.shape[1]))[:,0,:,:]
        return np.mean(sw, axis=1)

    def fit(self, t, y, u=None):
        # 1) (Optional) smoothing, then numerical derivative
        y_use = self._smooth(y, self.smooth_window).astype(float)
        dXdt = finite_difference(y_use, t)

        # 2) Build and scale library
        Theta = build_library(y_use, self.poly_order, self.include_sin_cos)
        Theta_s, scale = scale_columns(Theta)
        self.scale = scale

        # 3) STLSQ with ridge
        Xi = stlsq(Theta_s, dXdt, lam=self.lam, max_iter=self.max_iter, ridge=self.ridge)
        self.Xi = Xi  # coefficients in the *scaled* feature space

    def _phi_row(self, x_row):
        """Evaluate one feature row and apply training-time column scaling."""
        x = x_row[None,:]
        Theta_row = build_library(x, self.poly_order, self.include_sin_cos)[0]
        return Theta_row / (self.scale if self.scale is not None else 1.0)

    def predict_derivative(self, t, x, u=None):
        if self.Xi is None:
            return np.zeros_like(x)
        theta = self._phi_row(x)             # scaled features
        dxdt = theta @ self.Xi               # linear combination
        # Guard against NaN/Inf in pathological cases
        return np.nan_to_num(dxdt, nan=0.0, posinf=1e6, neginf=-1e6)
