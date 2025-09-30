import numpy as np
from .base import Model, register_model
from .sindy_stlsq import build_library

def build_integral_system(t, y, win_m=3):
    """Construct integral equations:
       Δx_i = ∫ Θ(x) dt * Xi  over sliding windows of length win_m.
       Returns S (integrated library) and dX (state increments).
    """
    N, d = y.shape
    if N <= win_m:
        raise ValueError("Not enough samples for integral windows")
    # precompute dt
    dt = np.diff(t)
    # library at nodes
    Theta_all, names = build_library(y, poly_order=3, include_sin_cos=False)  # order/opts will be overwritten by caller if needed
    # trapezoid integration over windows
    rows = []
    dX = []
    for i in range(N - win_m):
        j = i + win_m
        # Δx
        dX.append(y[j] - y[i])
        # integrate Theta over [i, j] using trapezoid per interval
        Sij = np.zeros((Theta_all.shape[1], d))
        # but we need S independent of d; compute single row (Theta integral)
        integ = np.zeros(Theta_all.shape[1])
        for k in range(i, j):
            dt_k = t[k+1] - t[k]
            integ += 0.5 * (Theta_all[k] + Theta_all[k+1]) * dt_k
        rows.append(integ)
    S = np.stack(rows, axis=0)   # shape [M, n_features]
    dX = np.stack(dX, axis=0)    # shape [M, d]
    return S, dX, names

@register_model
class SINDyPI(Model):
    name = "sindy_pi"
    def __init__(self, poly_order=3, include_sin_cos=False, lam=0.1, win_m=3):
        super().__init__(poly_order=poly_order, include_sin_cos=include_sin_cos, lam=lam, win_m=win_m)
        self.poly_order = poly_order
        self.include_sin_cos = include_sin_cos
        self.lam = lam
        self.win_m = win_m
        self.Xi = None
        self._eval_theta = None

    def fit(self, t, y, u=None):
        # rebuild library with requested options inside integral builder
        # We'll copy build_library here for consistency
        from .sindy_stlsq import build_library as _build_lib
        Theta_all, _ = _build_lib(y, self.poly_order, self.include_sin_cos)
        # construct integrals
        N = len(t)
        dt = np.diff(t)
        rows = []
        dX = []
        for i in range(N - self.win_m):
            j = i + self.win_m
            dX.append(y[j] - y[i])
            integ = np.zeros(Theta_all.shape[1])
            for k in range(i, j):
                dt_k = t[k+1] - t[k]
                integ += 0.5 * (Theta_all[k] + Theta_all[k+1]) * dt_k
            rows.append(integ)
        S = np.stack(rows, axis=0)
        dX = np.stack(dX, axis=0)
        # Solve per-dimension with thresholded LS
        d = y.shape[1]
        Xi = np.zeros((S.shape[1], d))
        for j in range(d):
            cj = np.linalg.lstsq(S, dX[:,j], rcond=None)[0]
            mask = np.abs(cj) >= self.lam
            if np.any(mask):
                cj_ref = np.zeros_like(cj)
                cj_ref[mask] = np.linalg.lstsq(S[:,mask], dX[:,j], rcond=None)[0]
                cj = cj_ref
            Xi[:,j] = cj
        self.Xi = Xi
        self._eval_theta = lambda x: _build_lib(x[None,:], self.poly_order, self.include_sin_cos)[0][0]

    def predict_derivative(self, t, x, u=None):
        theta = self._eval_theta(x)
        return theta @ self.Xi
