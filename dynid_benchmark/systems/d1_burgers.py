import numpy as np
from .base import DynamicalSystem

class Burgers1D(DynamicalSystem):
    def simulate_true(self, T: float, dt_true: float, seed=None):
        p = self.params
        nu = p.get("nu", 0.01/np.pi)
        L = p.get("L", 1.0)
        Nx = p.get("Nx", 128)

        x = np.linspace(0, L, Nx, endpoint=False)
        u0 = np.sin(2*np.pi*x/L) + 0.5*np.sin(np.pi*x/L)

        # Periodic BC with central differences
        dx = L / Nx
        def dudx(u):
            return (np.roll(u,-1) - np.roll(u,1)) / (2*dx)
        def d2udx2(u):
            return (np.roll(u,-1) - 2*u + np.roll(u,1)) / (dx*dx)

        u = u0.copy()
        t = 0.0
        ts = [0.0]
        Us = [u.copy()]
        Nsteps = int(np.floor(T/dt_true))
        for _ in range(Nsteps):
            def rhs(u_):
                return -u_ * dudx(u_) + nu * d2udx2(u_)
            u1 = u + dt_true * rhs(u)
            u2 = 0.75*u + 0.25*(u1 + dt_true*rhs(u1))
            u  = (1.0/3.0)*u + (2.0/3.0)*(u2 + dt_true*rhs(u2))
            t += dt_true
            ts.append(t); Us.append(u.copy())
        return {"t": np.array(ts), "x": np.stack(Us, axis=0)}
