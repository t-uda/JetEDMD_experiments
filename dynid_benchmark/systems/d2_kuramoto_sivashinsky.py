import numpy as np
from .base import DynamicalSystem

class KuramotoSivashinsky(DynamicalSystem):
    def simulate_true(self, T: float, dt_true: float, seed=None):
        p = self.params
        L = p.get("L", 22.0)
        Nx = p.get("Nx", 128)
        rng = np.random.default_rng(seed)

        # Spectral grid
        x = np.linspace(0, L, Nx, endpoint=False)
        k = 2*np.pi*np.fft.fftfreq(Nx, d=L/Nx)
        ik = 1j*k
        Lk = -(k**2) - (k**4)  # linear operator in Fourier space

        # ETDRK4 coefficients
        E = np.exp(Lk*dt_true)
        E2 = np.exp(Lk*dt_true/2.0)
        M = 16
        r = np.exp(1j*np.pi*(np.arange(1,M+1)-0.5)/M)
        Lk_mat = Lk[:,None] * np.ones((Nx,M))
        # Avoid singularities with small eps in denominators
        eps = 1e-14
        Q = dt_true * np.mean( (np.exp(Lk_mat* (dt_true/2.0) ) - 1.0) / (Lk_mat + eps), axis=1 )
        f1 = dt_true * np.mean( (-4 - Lk_mat*dt_true + np.exp(Lk_mat*dt_true)*(4 - 3*Lk_mat*dt_true + (Lk_mat*dt_true)**2)) / (Lk_mat**3 + eps), axis=1 )
        f2 = dt_true * np.mean( (2 + Lk_mat*dt_true + np.exp(Lk_mat*dt_true)*(-2 + Lk_mat*dt_true)) / (Lk_mat**3 + eps), axis=1 )
        f3 = dt_true * np.mean( (-4 - 3*Lk_mat*dt_true - (Lk_mat*dt_true)**2 + np.exp(Lk_mat*dt_true)*(4 - Lk_mat*dt_true)) / (Lk_mat**3 + eps), axis=1 )

        # Initial condition: small random field
        u = 0.1*rng.standard_normal(Nx)
        v = np.fft.fft(u)
        t = 0.0
        ts = [0.0]
        Us = [u.copy()]

        Nsteps = int(np.floor(T/dt_true))
        for _ in range(Nsteps):
            Nv = -0.5j*k*np.fft.fft(np.real(np.fft.ifft(v))**2)
            a = E2*v + Q*Nv
            Na = -0.5j*k*np.fft.fft(np.real(np.fft.ifft(a))**2)
            b = E2*v + Q*Na
            Nb = -0.5j*k*np.fft.fft(np.real(np.fft.ifft(b))**2)
            c = E2*a + Q*(2*Nb - Nv)
            Nc = -0.5j*k*np.fft.fft(np.real(np.fft.ifft(c))**2)
            v = E*v + (f1*Nv + 2*f2*(Na+Nb) + f3*Nc)
            u = np.real(np.fft.ifft(v))
            t += dt_true
            ts.append(t); Us.append(u.copy())

        return {"t": np.array(ts), "x": np.stack(Us, axis=0)}
