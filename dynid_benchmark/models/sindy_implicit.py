import numpy as np
from .base import Model, register_model
from .sindy_stlsq import build_library, finite_difference

@register_model
class ImplicitSINDy(Model):
    """Simplified implicit-SINDy:
    - Build augmented library Ψ = [Θ(X) , dXdt_j]
    - Find approximate null vector via SVD (smallest singular vector)
    - Convert to explicit form and refine with thresholded LS.
    """
    name = "sindy_implicit"

    def __init__(self, poly_order=3, include_sin_cos=False, lam=0.1):
        super().__init__(poly_order=poly_order, include_sin_cos=include_sin_cos, lam=lam)
        self.poly_order = poly_order
        self.include_sin_cos = include_sin_cos
        self.lam = lam
        self.Xi = None
        self._eval_theta = None

    def fit(self, t, y, u=None):
        dXdt = finite_difference(y, t)
        Theta, names = build_library(y, self.poly_order, self.include_sin_cos)
        d = y.shape[1]
        Xi = np.zeros((Theta.shape[1], d))
        for j in range(d):
            Psi = np.column_stack([Theta, dXdt[:,j]])
            # SVD-based null vector
            U,S,Vt = np.linalg.svd(Psi, full_matrices=False)
            v = Vt[-1]  # right singular vector for smallest singular value
            coeff_d = v[-1]
            if np.abs(coeff_d) < 1e-10:
                # fallback: explicit LS
                cj = np.linalg.lstsq(Theta, dXdt[:,j], rcond=None)[0]
            else:
                cj = -v[:-1] / coeff_d  # dXdt ≈ Θ c
            # threshold refine
            mask = np.abs(cj) >= self.lam
            if np.any(mask):
                cj_ref = np.zeros_like(cj)
                cj_ref[mask] = np.linalg.lstsq(Theta[:,mask], dXdt[:,j], rcond=None)[0]
                cj = cj_ref
            Xi[:,j] = cj
        self.Xi = Xi
        self._eval_theta = lambda x: build_library(x[None,:], self.poly_order, self.include_sin_cos)[0][0]

    def predict_derivative(self, t, x, u=None):
        theta = self._eval_theta(x)
        return theta @ self.Xi
