import numpy as np
from .base import Model, register_model


def build_poly(y, order=3, include_const=True):
    N, d = y.shape
    feats = []
    if include_const:
        feats.append(np.ones((N, 1)))
    from itertools import combinations_with_replacement

    names = []
    for p in range(1, order + 1):
        for combo in combinations_with_replacement(range(d), p):
            term = np.prod([y[:, j] for j in combo], axis=0)[:, None]
            feats.append(term)
            names.append("*".join([f"x{j}" for j in combo]))
    Theta = np.concatenate(feats, axis=1) if feats else np.empty((N, 0))
    return Theta, names


@register_model
class ImplicitSINDy(Model):
    """
    Simplified implicit-SINDy: Solve nullspace of [Theta(X) | dXdt_j] v ≈ 0
    for each component j, then convert to explicit form if possible:
      v = [xi; alpha], derivative ≈ - (Theta xi) / alpha
    """

    name = "implicit_sindy"

    def __init__(self, order=3, include_const=True, eps=1e-8):
        super().__init__(order=order, include_const=include_const, eps=eps)
        self.order = order
        self.include_const = include_const
        self.eps = eps
        self.Xi = None
        self.Theta_names = None

    def fit(self, t, y, u=None):
        # numerical derivative is still used, but implicit solves a homogeneous system
        N, d = y.shape
        dt = np.diff(t)
        dY = np.zeros_like(y)
        dY[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])[:, None]
        dY[0] = (y[1] - y[0]) / dt[0]
        dY[-1] = (y[-1] - y[-2]) / dt[-1]
        Theta, names = build_poly(y, self.order, self.include_const)
        nfeat = Theta.shape[1]
        Xi = np.zeros((nfeat, d))
        for j in range(d):
            A = np.concatenate([Theta, dY[:, j : j + 1]], axis=1)  # (N, nfeat+1)
            # SVD: smallest right singular vector
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            v = Vt[-1]  # (nfeat+1,)
            alpha = v[-1]
            if abs(alpha) < self.eps:
                # fallback to explicit regression
                beta, *_ = np.linalg.lstsq(Theta, dY[:, j], rcond=None)
                Xi[:, j] = beta
            else:
                Xi[:, j] = -v[:-1] / alpha
        self.Xi = Xi
        self.Theta_names = names

    def predict_derivative(self, t, x, u=None):
        xr = x[None, :]
        Theta_x, _ = build_poly(xr, self.order, self.include_const)
        return (Theta_x @ self.Xi).ravel()
