# PyTorch を用いた最小限の Neural ODE ベースライン（任意依存性）
# torch が利用できない場合は fit() 時に明示的なエラーを送出
import numpy as np
from .base import Model, register_model

try:
    import torch
    import torch.nn as nn

    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    _TORCH_ERR = e

if TORCH_OK:

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden=128, depth=2, out_dim=None):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(depth):
                layers.append(nn.Linear(d, hidden))
                layers.append(nn.Tanh())
                d = hidden
            layers.append(nn.Linear(d, out_dim if out_dim is not None else in_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

else:  # pragma: no cover - torch が使えない環境のみで実行

    class MLP:  # type: ignore[override]
        def __init__(self, *_, **__):
            raise RuntimeError(
                "PyTorch is required for NeuralODE but could not be imported"
            )


@register_model
class NeuralODETorch(Model):
    name = "neural_ode"

    def __init__(
        self, hidden=128, depth=2, lr=1e-3, iters=2000, weight_decay=1e-6, batch=0
    ):
        super().__init__(
            hidden=hidden,
            depth=depth,
            lr=lr,
            iters=iters,
            weight_decay=weight_decay,
            batch=batch,
        )
        self.hidden = hidden
        self.depth = depth
        self.lr = lr
        self.iters = iters
        self.weight_decay = weight_decay
        self.batch = batch  # 0 の場合は全データを一括学習
        self.model = None
        self.with_input = False

    def fit(self, t, y, u=None):
        if not TORCH_OK:
            raise RuntimeError(
                f"PyTorch is required for NeuralODE but not available: {_TORCH_ERR}"
            )
        self.with_input = u is not None and len(u) == len(y)
        # 学習デバイスを自動選択し、入力次元を系の状態＋入力で設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = y.shape[1]
        in_dim = d + (u.shape[1] if self.with_input else 0)
        self.model = MLP(in_dim, hidden=self.hidden, depth=self.depth, out_dim=d).to(
            device
        )
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        tt = torch.tensor(t, dtype=torch.float32, device=device)
        yy = torch.tensor(y, dtype=torch.float32, device=device)
        uu = (
            torch.tensor(u, dtype=torch.float32, device=device)
            if self.with_input
            else None
        )

        def f(ti, xi, ui=None):
            if ui is None:
                inp = xi
            else:
                inp = torch.cat([xi, ui], dim=-1)
            return self.model(inp)

        def rollout_times(tt, x0, uu):
            # PyTorch テンソル上で RK4 により時間発展を計算
            x = [x0]
            for i in range(1, len(tt)):
                dt = tt[i] - tt[i - 1]
                xi = x[-1]
                ui = uu[i - 1 : i] if uu is not None else None
                k1 = f(tt[i - 1], xi, ui)
                k2 = f(tt[i - 1] + 0.5 * dt, xi + 0.5 * dt * k1, ui)
                k3 = f(tt[i - 1] + 0.5 * dt, xi + 0.5 * dt * k2, ui)
                k4 = f(tt[i - 1] + dt, xi + dt * k3, ui)
                # 4 次の RK 法でニューラルネットが定義するベクトル場を積分
                x_next = xi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                x.append(x_next)
            return torch.stack(x, dim=0)

        x0 = yy[0]
        for it in range(self.iters):
            opt.zero_grad()
            yhat = rollout_times(tt, x0, uu)
            loss = torch.mean((yhat - yy) ** 2)
            loss.backward()
            opt.step()
            # TODO: ここで patience 付き早期終了などを検討すると良い

        self._device = device
        self._tt = tt  # not strictly needed

    def predict_derivative(self, t, x, u=None):
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        # torch.no_grad 下で 1 ステップ分のベクトル場を推論
        import torch

        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x[None, :], dtype=torch.float32, device=self._device)
            if u is not None:
                ut = torch.tensor(u[None, :], dtype=torch.float32, device=self._device)
                inp = torch.cat([xt, ut], dim=-1)
            else:
                inp = xt
            dx = self.model(inp).cpu().numpy()[0]
        return dx
