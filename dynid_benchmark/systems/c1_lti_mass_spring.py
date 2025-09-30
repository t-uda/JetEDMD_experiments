import numpy as np
from .base import DynamicalSystem

def prbs(t, bit_len=0.2, amp=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return amp * (1 if int(t/bit_len) % 2 == 0 else -1)

def sine_sweep(t, f0=0.2, f1=2.0, T=60.0, amp=0.5):
    # linear chirp frequency sweep
    k = (f1 - f0) / T
    phase = 2*np.pi*(f0*t + 0.5*k*t*t)
    return amp*np.sin(phase)

def chirp(t, f0=0.2, f1=2.0, T=60.0, amp=0.5):
    return sine_sweep(t, f0, f1, T, amp)

class MassSpringInput(DynamicalSystem):
    def simulate_true(self, T: float, dt_true: float, seed=None):
        p = self.params
        zeta = p.get("zeta", 0.02)
        omega = p.get("omega", 2*np.pi)
        b = p.get("b", 1.0)
        mode = p.get("input_mode", "prbs")
        prbs_bit = p.get("prbs_bit", 0.2)
        amp = p.get("amp", 0.5)

        def u_fn(t, x):
            if mode == "prbs":
                return np.array([prbs(t, bit_len=prbs_bit, amp=amp, seed=seed or 0)])
            elif mode == "sine":
                return np.array([np.sin(2*np.pi*1.0*t)*amp])
            elif mode == "sweep":
                return np.array([sine_sweep(t, 0.2, 2.0, T, amp)])
            elif mode == "chirp":
                return np.array([chirp(t, 0.2, 2.0, T, amp)])
            else:
                return np.array([0.0])

        def f(t, s, u):
            x, v = s
            a = -2*zeta*omega*v - (omega**2)*x + b*u[0]
            return np.array([v, a])

        from .base import simulate_ode
        x0 = np.array([0.0, 0.0])
        return simulate_ode(f, x0, T, dt_true, with_u=True, u_fn=u_fn)
