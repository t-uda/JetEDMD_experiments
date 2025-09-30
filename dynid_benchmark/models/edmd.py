import numpy as np
from .base import Model, register_model

def poly_lift(x, order=2, include_const=True):
    # x: (N, d)
    N, d = x.shape
    Phi = []
    names = []
    if include_const:
        Phi.append(np.ones((N,1))); names.append("1")
    # linear
    for j in range(d):
        Phi.append(x[:,j:j+1]); names.append(f"x{j}")
    if order >= 2:
        from itertools import combinations_with_replacement
        for p in range(2, order+1):
            for combo in combinations_with_replacement(range(d), p):
                term = np.prod([x[:,j] for j in combo], axis=0)[:,None]
                Phi.append(term)
                names.append("*".join([f"x{j}" for j in combo]))
    Phi = np.concatenate(Phi, axis=1)
    return Phi, names

@register_model
class EDMD(Model):
    """Discrete-time EDMD with polynomial dictionary.
    Note: treats data as uniformly sampled; ignores inputs by default.
    """
    name = "edmd"

    def __init__(self, order=2, ridge=1e-6):
        super().__init__(order=order, ridge=ridge)
        self.order = order
        self.ridge = ridge
        self.K = None
        self.C = None
        self._phi = None

    def fit(self, t, y, u=None):
        # Build snapshot pairs (k -> k+1)
        X = y[:-1,:]
        Xp = y[1:,:]
        Z, _ = poly_lift(X, order=self.order)
        Zp, _ = poly_lift(Xp, order=self.order)
        # Solve Z K ≈ Zp  (ridge)
        lam = self.ridge
        self.K = np.linalg.lstsq(Z.T@Z + lam*np.eye(Z.shape[1]), Z.T@Zp, rcond=None)[0]
        # Decoder: y ≈ Z C
        self.C = np.linalg.lstsq(Z, X, rcond=None)[0]
        self._phi = lambda x: poly_lift(x[None,:], order=self.order)[0][0]

    def predict_derivative(self, t, x, u=None):
        # EDMD is discrete; derivative not defined. Provide zero (unused).
        return np.zeros_like(x)

    def rollout(self, t, x0, u=None):
        # Discrete evolution per time step in t
        n_steps = len(t)
        Y = np.zeros((n_steps, len(x0)), dtype=float)
        z = self._phi(x0)  # lifted
        Y[0] = x0
        for k in range(1, n_steps):
            z = z @ self.K
            xk = z @ self.C
            Y[k] = xk
        return Y
