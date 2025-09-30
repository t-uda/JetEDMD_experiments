import numpy as np
from .base import Model, register_model
from .sindy_stlsq import build_library, finite_difference


def scale_columns(arr: np.ndarray, eps: float = 1e-12):
    norms = np.linalg.norm(arr, axis=0)
    norms = np.where(norms < eps, 1.0, norms)
    return arr / norms, norms  # 基底ベクトルを正規化してスケール不変性を確保


@register_model
class ImplicitSINDy(Model):
    """列正規化と SVD による零空間推定を組み合わせた implicit SINDy"""

    name = "sindy_implicit"

    def __init__(
        self,
        poly_order: int = 3,
        denom_order: int = 1,
        include_sin_cos: bool = False,
        thresh: float = 1e-3,
    ):
        super().__init__(
            poly_order=poly_order,
            denom_order=denom_order,
            include_sin_cos=include_sin_cos,
            thresh=thresh,
        )
        self.poly_order = poly_order
        self.denom_order = denom_order
        self.include_sin_cos = include_sin_cos
        self.thresh = thresh
        self.coeffs = None
        self._scale_num = None
        self._scale_den = None

    def fit(self, t, y, u=None):
        dXdt = finite_difference(y, t)
        theta_num, _ = build_library(y, self.poly_order, self.include_sin_cos)
        theta_den, _ = build_library(y, max(self.denom_order, 0), self.include_sin_cos)
        # 分子側と分母側のライブラリを個別に構築し、各基底を正規化
        theta_num_s, scale_num = scale_columns(theta_num)
        theta_den_s, scale_den = scale_columns(theta_den)
        self._scale_num = scale_num
        self._scale_den = scale_den

        d = y.shape[1]
        coeffs = []
        for j in range(d):
            augmented = np.concatenate(
                [theta_num_s, theta_den_s * dXdt[:, j : j + 1]],
                axis=1,
            )
            # 拡張行列の零空間を特異値分解で抽出
            _, _, vh = np.linalg.svd(augmented, full_matrices=False)
            v = vh[-1]
            v[np.abs(v) < self.thresh] = 0.0
            num = v[: theta_num_s.shape[1]]
            den = v[theta_num_s.shape[1] :]
            if np.linalg.norm(den) < 1e-10:
                beta, *_ = np.linalg.lstsq(theta_num_s, dXdt[:, j], rcond=None)
                coeffs.append(("explicit", beta))
            else:
                coeffs.append(("implicit", (num, den)))
        self.coeffs = coeffs

    def _phi_row(self, x: np.ndarray, order: int, scale: np.ndarray) -> np.ndarray:
        theta_x, _ = build_library(x[None, :], order, self.include_sin_cos)
        row = theta_x[0]
        if scale is not None:
            row = row / scale  # 学習時のスケーリングと整合するよう調整
        return row

    def predict_derivative(self, t, x, u=None):
        if self.coeffs is None:
            raise RuntimeError("Implicit SINDy model is not fitted yet")
        psi_x = self._phi_row(x, self.poly_order, self._scale_num)
        phi_x = self._phi_row(x, max(self.denom_order, 0), self._scale_den)
        out = np.zeros_like(x, dtype=float)
        for j, (mode, params) in enumerate(self.coeffs):
            if mode == "explicit":
                out[j] = psi_x @ params
            else:
                num, den = params
                numerator = psi_x @ num
                denominator = phi_x @ den
                if not np.isfinite(denominator) or abs(denominator) < 1e-8:
                    denominator = 1e-8 if denominator >= 0 else -1e-8
                out[j] = -numerator / denominator  # 暗黙的表示を陽的な微分に変換
        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
