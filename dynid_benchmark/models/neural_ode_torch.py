# Minimal Neural ODE baseline using PyTorch (optional dependency).
# If torch is not available, the model will raise a clear error on fit().
import numpy as np
from .base import Model, register_model

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    _TORCH_ERR = e

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

@register_model
class NeuralODETorch(Model):
    name = "neural_ode"

    def __init__(self, hidden=128, depth=2, lr=1e-3, iters=2000, weight_decay=1e-6, batch=0):
        super().__init__(hidden=hidden, depth=depth, lr=lr, iters=iters, weight_decay=weight_decay, batch=batch)
        self.hidden = hidden; self.depth = depth
        self.lr = lr; self.iters = iters; self.weight_decay = weight_decay
        self.batch = batch  # 0 = full-batch
        self.model = None
        self.with_input = False

    def fit(self, t, y, u=None):
        if not TORCH_OK:
            raise RuntimeError(f"PyTorch is required for NeuralODE but not available: {_TORCH_ERR}")
        self.with_input = (u is not None and len(u) == len(y))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = y.shape[1]
        in_dim = d + (u.shape[1] if self.with_input else 0)
        self.model = MLP(in_dim, hidden=self.hidden, depth=self.depth, out_dim=d).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        tt = torch.tensor(t, dtype=torch.float32, device=device)
        yy = torch.tensor(y, dtype=torch.float32, device=device)
        uu = torch.tensor(u, dtype=torch.float32, device=device) if self.with_input else None

        def f(ti, xi, ui=None):
            if ui is None:
                inp = xi
            else:
                inp = torch.cat([xi, ui], dim=-1)
            return self.model(inp)

        def rollout_times(tt, x0, uu):
            x = [x0]
            for i in range(1, len(tt)):
                dt = (tt[i] - tt[i-1])
                xi = x[-1]
                ui = uu[i-1:i] if uu is not None else None
                k1 = f(tt[i-1], xi, ui)
                k2 = f(tt[i-1]+0.5*dt, xi + 0.5*dt*k1, ui)
                k3 = f(tt[i-1]+0.5*dt, xi + 0.5*dt*k2, ui)
                k4 = f(tt[i-1]+dt, xi + dt*k3, ui)
                x_next = xi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                x.append(x_next)
            return torch.stack(x, dim=0)

        x0 = yy[0]
        for it in range(self.iters):
            opt.zero_grad()
            yhat = rollout_times(tt, x0, uu)
            loss = torch.mean((yhat - yy)**2)
            loss.backward()
            opt.step()
            # (optional) could add early-stopping with patience

        self._device = device
        self._tt = tt  # not strictly needed

    def predict_derivative(self, t, x, u=None):
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        # one-step derivative on CPU using torch.no_grad
        import torch
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x[None,:], dtype=torch.float32, device=self._device)
            if u is not None:
                ut = torch.tensor(u[None,:], dtype=torch.float32, device=self._device)
                inp = torch.cat([xt, ut], dim=-1)
            else:
                inp = xt
            dx = self.model(inp).cpu().numpy()[0]
        return dx
