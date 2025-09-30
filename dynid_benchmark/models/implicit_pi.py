import numpy as np
from .base import Model, register_model

def build_poly(y, order=3):
    """Polynomial library with constant term."""
    N, d = y.shape
    Phi = [np.ones((N,1))]
    from itertools import combinations_with_replacement
    for p in range(1, order+1):
        for combo in combinations_with_replacement(range(d), p):
            Phi.append(np.prod([y[:,j] for j in combo], axis=0)[:,None])
    return np.concatenate(Phi, axis=1)

def finite_difference(y, t):
    """Central finite difference with one-sided edges."""
    dy = np.zeros_like(y, dtype=float)
    dt = np.diff(t); dt = np.clip(dt, 1e-12, None)
    dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])[:,None]
    dy[0]     = (y[1]  - y[0])   / dt[0]
    dy[-1]    = (y[-1] - y[-2])  / dt[-1]
    return dy

def scale_columns(A, eps=1e-12):
    """Column L2 scaling for conditioning."""
    s = np.linalg.norm(A, axis=0)
    s = np.where(s < eps, 1.0, s)
    return A / s, s

def stlsq(Theta, Y, lam=0.1, iters=10, ridge=1e-8):
    """STLSQ with ridge (for SINDy-PI)."""
    G = Theta.T @ Theta + ridge*np.eye(Theta.shape[1])
    Xi = np.linalg.solve(G, Theta.T @ Y)
    for _ in range(iters):
        small = np.abs(Xi) < lam
        Xi[small] = 0.0
        for k in range(Y.shape[1]):
            idx = ~small[:,k]
            if np.sum(idx)==0: 
                continue
            Gk = Theta[:,idx].T @ Theta[:,idx] + ridge*np.eye(np.sum(idx))
            Xi[idx,k] = np.linalg.solve(Gk, Theta[:,idx].T @ Y[:,k])
    return Xi

@register_model
class SINDyPI(Model):
    """SINDy-PI (integral formulation). Stabilized by library scaling + ridge."""
    name = "sindy_pi"

    def __init__(self, poly_order=3, window_len=3, lam=0.1, max_iter=10, ridge=1e-8):
        super().__init__(poly_order=poly_order, window_len=window_len, lam=lam, max_iter=max_iter, ridge=ridge)
        self.poly_order = poly_order
        self.window_len = int(window_len)
        self.lam = lam
        self.max_iter = max_iter
        self.ridge = ridge
        self.Xi = None
        self.scale = None

    def fit(self, t, y, u=None):
        Phi_all = build_poly(y, self.poly_order)
        Phi_s, scale = scale_columns(Phi_all)
        self.scale = scale

        N = len(t); w = self.window_len
        rows = []; targets = []
        # Build integral design matrix and targets Δx
        for i in range(0, N - w, 1):
            dx = y[i+w] - y[i]
            S = np.zeros(Phi_s.shape[1])
            for k in range(i, i+w):
                S += Phi_s[k] * (t[k+1] - t[k])
            rows.append(S); targets.append(dx)
        Theta = np.stack(rows, axis=0)
        Y = np.stack(targets, axis=0)
        # STLSQ with ridge
        self.Xi = stlsq(Theta, Y, lam=self.lam, iters=self.max_iter, ridge=self.ridge)

    def _phi(self, x):
        return (build_poly(x[None,:], self.poly_order)[0]) / (self.scale if self.scale is not None else 1.0)

    def predict_derivative(self, t, x, u=None):
        # PI gives Δx ≈ (∫ Phi dt) Xi. For instantaneous derivative,
        # approximate by Phi(x) @ Xi (heuristic consistent with small windows).
        dxdt = self._phi(x) @ self.Xi
        return np.nan_to_num(dxdt, nan=0.0, posinf=1e6, neginf=-1e6)

@register_model
class ImplicitSINDy(Model):
    """Implicit-SINDy (simplified).
    We compute a homogeneous relation per state dimension
      0 ≈ [Psi(x), Phi(x) ⊙ xdot_j] · v
    via SVD (smallest singular vector). We then threshold coefficients
    and derive a rational form: xdot_j = - (Psi a) / (Phi b).
    Columns are scaled for conditioning; we avoid re-solving ill-posed
    zero-residual problems that can produce NaN.
    """
    name = "implicit_sindy"

    def __init__(self, poly_order_x=3, poly_order_v=1, thresh=1e-3):
        super().__init__(poly_order_x=poly_order_x, poly_order_v=poly_order_v, thresh=thresh)
        self.px = poly_order_x
        self.pv = poly_order_v
        self.thresh = thresh
        self.coeffs = None  # list of ("explicit", W) or ("implicit", (a,b))
        self.scale_Psi = None
        self.scale_Phi = None

    def fit(self, t, y, u=None):
        ydot = finite_difference(y, t)
        N, d = y.shape
        Psi = build_poly(y, self.px)
        Phi = build_poly(y, self.pv)
        Psi_s, sPsi = scale_columns(Psi); Phi_s, sPhi = scale_columns(Phi)
        self.scale_Psi, self.scale_Phi = sPsi, sPhi

        self.coeffs = []
        for j in range(d):
            # Θ_j = [Psi_s, (Phi_s ⊙ ydot_j)]
            Theta = np.concatenate([Psi_s, Phi_s * ydot[:,j:j+1]], axis=1)
            # Smallest right singular vector (nullspace approximation)
            U, S, Vh = np.linalg.svd(Theta, full_matrices=False)
            v = Vh[-1]
            # Simple hard threshold to enforce sparsity; avoid re-solving to zero
            v[np.abs(v) < self.thresh] = 0.0
            a = v[:Psi_s.shape[1]]
            b = v[Psi_s.shape[1]:]
            # Fallback: if denominator ~0, revert to explicit LS on Psi
            if np.linalg.norm(b) < 1e-10:
                W = np.linalg.lstsq(Psi_s, ydot[:,j], rcond=None)[0]
                self.coeffs.append(("explicit", W))
            else:
                self.coeffs.append(("implicit", (a, b)))

    def _phi_row(self, x, order, scale):
        return build_poly(x[None,:], order)[0] / (scale if scale is not None else 1.0)

    def predict_derivative(self, t, x, u=None):
        d = len(x)
        out = np.zeros(d, dtype=float)
        Psi_x = self._phi_row(x, self.px, self.scale_Psi)
        Phi_x = self._phi_row(x, self.pv, self.scale_Phi)
        for j in range(d):
            tag, par = self.coeffs[j]
            if tag == "explicit":
                W = par
                out[j] = Psi_x @ W
            else:
                a, b = par
                num = Psi_x @ a
                den = Phi_x @ b
                # Guard against tiny denominators
                if not np.isfinite(den) or abs(den) < 1e-8:
                    den = 1e-8 if den >= 0 else -1e-8
                out[j] = - num / den
        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
