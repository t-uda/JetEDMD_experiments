import numpy as np
from .base import Model, register_model


def poly_lift(x, order=2, include_const=True):
    # x: (N, d)
    N, d = x.shape
    Phi = []
    names = []
    if include_const:
        Phi.append(np.ones((N, 1)))
        names.append("1")
    # 1 次項の追加
    for j in range(d):
        Phi.append(x[:, j : j + 1])
        names.append(f"x{j}")
    if order >= 2:
        from itertools import combinations_with_replacement

        for p in range(2, order + 1):
            for combo in combinations_with_replacement(range(d), p):
                term = np.prod([x[:, j] for j in combo], axis=0)[:, None]
                Phi.append(term)
                names.append("*".join([f"x{j}" for j in combo]))
    Phi = np.concatenate(Phi, axis=1)
    return Phi, names


@register_model
class EDMD(Model):
    """多項式辞書を用いた離散時間 EDMD。データは等間隔サンプリングとみなし、入力は既定で無視する。"""

    name = "edmd"

    def __init__(
        self,
        order=2,
        ridge=0.0,
        solver: str = "svd",
        svd_tol: float = 1e-12,
        svd_rank: int | None = None,
    ):
        super().__init__(order=order, ridge=ridge)
        self.order = order
        self.ridge = ridge
        self.solver = solver
        self.svd_tol = svd_tol
        self.svd_rank = svd_rank
        self.K = None
        self.C = None
        self._phi = None

    def fit(self, t, y, u=None):
        # 連続する状態の組 (k -> k+1) を構成して離散時間の線形化を表現
        X = y[:-1, :]
        Xp = y[1:, :]
        Z, _ = poly_lift(X, order=self.order)
        Zp, _ = poly_lift(Xp, order=self.order)
        # コーシャン演算子 K を求める（条件に応じて解法を切替）
        lam = float(self.ridge)
        solver = self.solver
        if solver not in {"auto", "normal", "svd"}:
            raise ValueError("solver must be 'auto', 'normal', or 'svd'")

        use_svd = solver == "svd" or (solver == "auto" and lam == 0.0)
        if use_svd:
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            if self.svd_rank is not None:
                rank = min(self.svd_rank, len(S))
            else:
                thresh = self.svd_tol * S[0] if len(S) else 0.0
                rank = int(np.sum(S > thresh))
                if rank == 0 and len(S):
                    rank = 1
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vt_r = Vt[:rank, :]
            if lam > 0.0:
                # リッジありの場合は S/(S^2 + lam) の縮小係数を適用
                shrink = S_r / (S_r**2 + lam)
            else:
                shrink = 1.0 / S_r
            Z_pinv = (Vt_r.T * shrink) @ U_r.T
            self.K = Z_pinv @ Zp
            self.C = Z_pinv @ X
        else:
            I = np.eye(Z.shape[1])
            self.K = np.linalg.lstsq(Z.T @ Z + lam * I, Z.T @ Zp, rcond=None)[0]
            self.C = np.linalg.lstsq(Z, X, rcond=None)[0]
        self._phi = lambda x: poly_lift(x[None, :], order=self.order)[0][0]

    def predict_derivative(self, t, x, u=None):
        # 離散モデルのため時間微分は定義せずゼロを返す（上位互換のための実装）
        return np.zeros_like(x)

    def rollout(self, t, x0, u=None):
        # 与えられた時刻列をステップ数として解釈し、離散的に状態を伝搬
        n_steps = len(t)
        Y = np.zeros((n_steps, len(x0)), dtype=float)
        z = self._phi(x0)  # lifted
        Y[0] = x0
        for k in range(1, n_steps):
            z = z @ self.K
            xk = z @ self.C
            Y[k] = xk
        return Y
