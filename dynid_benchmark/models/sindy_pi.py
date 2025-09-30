import numpy as np
from .base import Model, register_model
from .sindy_stlsq import build_library


def scale_columns(arr: np.ndarray, eps: float = 1e-12):
    norms = np.linalg.norm(arr, axis=0)
    norms = np.where(norms < eps, 1.0, norms)
    return arr / norms, norms


def stlsq(theta: np.ndarray, targets: np.ndarray, lam: float, max_iter: int, ridge: float):
    gram = theta.T @ theta + ridge * np.eye(theta.shape[1])
    coeffs = np.linalg.solve(gram, theta.T @ targets)
    for _ in range(max_iter):
        mask_small = np.abs(coeffs) < lam
        coeffs[mask_small] = 0.0
        for col in range(targets.shape[1]):
            keep = ~mask_small[:, col]
            if np.sum(keep) == 0:
                continue
            sub_gram = theta[:, keep].T @ theta[:, keep] + ridge * np.eye(np.sum(keep))
            coeffs[keep, col] = np.linalg.solve(sub_gram, theta[:, keep].T @ targets[:, col])
    return coeffs


@register_model
class SINDyPI(Model):
    """Integral-form SINDy with column scaling and ridge-stabilised STLSQ."""

    name = "sindy_pi"

    def __init__(
        self,
        poly_order: int = 3,
        include_sin_cos: bool = False,
        lam: float = 0.1,
        window_len: int = 3,
        max_iter: int = 10,
        ridge: float = 1e-8,
    ):
        super().__init__(
            poly_order=poly_order,
            include_sin_cos=include_sin_cos,
            lam=lam,
            window_len=window_len,
            max_iter=max_iter,
            ridge=ridge,
        )
        self.poly_order = poly_order
        self.include_sin_cos = include_sin_cos
        self.lam = lam
        self.window_len = max(1, int(window_len))
        self.max_iter = max_iter
        self.ridge = ridge
        self.Xi = None
        self._scale = None

    def fit(self, t, y, u=None):
        theta_all, _ = build_library(y, self.poly_order, self.include_sin_cos)
        theta_scaled, scale = scale_columns(theta_all)
        self._scale = scale

        n = len(t)
        w = self.window_len
        if n <= w:
            raise ValueError("Not enough samples for the requested integration window")

        rows = []
        targets = []
        for i in range(n - w):
            j = i + w
            targets.append(y[j] - y[i])
            integ = np.zeros(theta_scaled.shape[1], dtype=float)
            for k in range(i, j):
                dt = t[k + 1] - t[k]
                integ += 0.5 * (theta_scaled[k] + theta_scaled[k + 1]) * dt
            rows.append(integ)

        if not rows:
            raise ValueError("Failed to build any integral constraints for SINDy-PI")

        theta_int = np.stack(rows, axis=0)
        delta_x = np.stack(targets, axis=0)
        self.Xi = stlsq(theta_int, delta_x, lam=self.lam, max_iter=self.max_iter, ridge=self.ridge)

    def _phi_row(self, x: np.ndarray) -> np.ndarray:
        theta_x, _ = build_library(x[None, :], self.poly_order, self.include_sin_cos)
        row = theta_x[0]
        if self._scale is not None:
            row = row / self._scale
        return row

    def predict_derivative(self, t, x, u=None):
        if self.Xi is None:
            raise RuntimeError("SINDy-PI model is not fitted yet")
        theta = self._phi_row(x)
        dx = theta @ self.Xi
        return np.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)
