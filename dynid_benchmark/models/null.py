import numpy as np
from .base import Model, register_model

@register_model
class MeanDerivativeModel(Model):
    """Learns mean derivative (finite diff) and uses constant vector field."""
    name = "mean_dfdx"

    def fit(self, t, y, u=None):
        dy = np.zeros_like(y)
        dt = np.diff(t)
        dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])[:,None]
        dy[0] = (y[1] - y[0]) / dt[0]
        dy[-1] = (y[-1] - y[-2]) / dt[-1]
        self.mu = np.mean(dy, axis=0)

    def predict_derivative(self, t, x, u=None):
        return self.mu
