"""
Evaluation utilities for system identification experiments.

Includes:
- Robust RMSE (nan/inf-safe)
- FRF (Bode) via Welch/CSD (SciPy if available; NumPy fallback)
- PSD via Welch (SciPy if available; NumPy fallback)
- Largest Lyapunov exponent (Rosenstein method)
- Event time detection and timing error (e.g., impact times)

All functions are self-contained and documented for clarity.
"""
from __future__ import annotations

import numpy as np
import json, os
from typing import Dict, Tuple, Optional

# Optional SciPy for spectral analysis (welch, csd)
try:
    from scipy.signal import welch, csd, get_window
    SCIPY_SIG_OK = True
except Exception:
    SCIPY_SIG_OK = False

# ----------------------------------------------------------------------------------
# Basic I/O and robust metrics
# ----------------------------------------------------------------------------------
def save_metrics(path: str, metrics: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def rmse_robust(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE that ignores NaN/Inf and avoids overflow warnings by clipping extremes."""
    diff = a - b
    diff = np.where(np.isfinite(diff), diff, np.nan)
    # Clip very large magnitudes to avoid overflow in squaring
    diff = np.clip(diff, -1e6, 1e6)
    m = np.nanmean(diff**2)
    if not np.isfinite(m):
        return float("inf")
    return float(np.sqrt(m))

# ----------------------------------------------------------------------------------
# Spectral: PSD and FRF (Bode)
# ----------------------------------------------------------------------------------
def _uniform_dt(t: np.ndarray) -> float:
    dt = np.mean(np.diff(t))
    return float(dt)

def psd_welch(t: np.ndarray, y: np.ndarray, nperseg: Optional[int]=None, noverlap: Optional[int]=None):
    """Compute PSD for each column of y using Welch method.
    Returns (f, Pxx) where Pxx has shape [len(f), d].
    If SciPy is unavailable, falls back to a simple periodogram average.
    """
    y = np.atleast_2d(y)
    if y.shape[0] < y.shape[1]:
        y = y.T
    d = y.shape[1]
    fs = 1.0/_uniform_dt(t)

    if SCIPY_SIG_OK:
        if nperseg is None:
            nperseg = min(1024, max(64, len(t)//8))
        if noverlap is None:
            noverlap = nperseg//2
        f = None
        P = []
        for j in range(d):
            fj, Pj = welch(y[:,j], fs=fs, nperseg=nperseg, noverlap=noverlap)
            if f is None: f = fj
            P.append(Pj)
        Pxx = np.stack(P, axis=1)
        return f, Pxx
    else:
        # Simple fallback: split into segments, average periodograms
        if nperseg is None:
            nperseg = min(1024, max(64, len(t)//8))
        if noverlap is None:
            noverlap = nperseg//2
        step = nperseg - noverlap
        f = np.fft.rfftfreq(nperseg, d=1/fs)
        P = np.zeros((len(f), d))
        count = 0
        for start in range(0, len(t)-nperseg+1, step):
            seg = y[start:start+nperseg]
            seg = seg - seg.mean(axis=0, keepdims=True)
            Y = np.fft.rfft(seg, axis=0)
            P += (np.abs(Y)**2)/nperseg
            count += 1
        if count > 0:
            P = P / count
        return f, P

def frf_welch(t: np.ndarray, y: np.ndarray, u: np.ndarray, nperseg: Optional[int]=None, noverlap: Optional[int]=None):
    """Estimate frequency response H(f) = S_yu / S_uu for each output dimension of y.
    Returns (f, H) where H has shape [len(f), d] as complex values.
    """
    y = np.atleast_2d(y); u = np.atleast_2d(u)
    if y.shape[0] < y.shape[1]: y = y.T
    if u.shape[0] < u.shape[1]: u = u.T
    d = y.shape[1]
    fs = 1.0/_uniform_dt(t)

    if SCIPY_SIG_OK:
        if nperseg is None:
            nperseg = min(1024, max(64, len(t)//8))
        if noverlap is None:
            noverlap = nperseg//2
        f = None
        Hcols = []
        for j in range(d):
            fj, Syu = csd(y[:,j], u[:,0], fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Suu = welch(u[:,0], fs=fs, nperseg=nperseg, noverlap=noverlap)
            if f is None: f = fj
            Hcols.append(Syu / (Suu + 1e-12))
        H = np.stack(Hcols, axis=1)
        return f, H
    else:
        # Fallback: build Welch manually for cross/auto spectra
        if nperseg is None:
            nperseg = min(1024, max(64, len(t)//8))
        if noverlap is None:
            noverlap = nperseg//2
        step = nperseg - noverlap
        f = np.fft.rfftfreq(nperseg, d=1/fs)
        Hcols = [np.zeros(len(f), dtype=complex) for _ in range(d)]
        Suu_acc = np.zeros(len(f))
        count = 0
        for start in range(0, len(t)-nperseg+1, step):
            seg_u = u[start:start+nperseg,0] - np.mean(u[start:start+nperseg,0])
            U = np.fft.rfft(seg_u)
            Suu = (U*np.conj(U)).real / nperseg
            Suu_acc += Suu
            for j in range(d):
                seg_y = y[start:start+nperseg,j] - np.mean(y[start:start+nperseg,j])
                Y = np.fft.rfft(seg_y)
                Syu = Y * np.conj(U) / nperseg
                Hcols[j] += Syu
            count += 1
        Suu_mean = Suu_acc / max(count,1)
        H = np.stack([Hcols[j] / (Suu_mean + 1e-12) for j in range(d)], axis=1)
        return f, H

def bode_metrics(f_true: np.ndarray, H_true: np.ndarray, f_hat: np.ndarray, H_hat: np.ndarray) -> Dict:
    """Compute Bode errors between true and estimated FRFs.
    Interpolate |H| and ∠H onto a common frequency grid (true's grid)."""
    # Interpolate H_hat onto f_true grid (real/imag separately)
    H_hat_i = np.interp(f_true, f_hat, np.real(H_hat)) + 1j*np.interp(f_true, f_hat, np.imag(H_hat))
    mag_true = np.abs(H_true); mag_hat = np.abs(H_hat_i)
    # Avoid log(0)
    mag_true = np.clip(mag_true, 1e-12, None); mag_hat = np.clip(mag_hat, 1e-12, None)
    mag_err = np.mean((20*np.log10(mag_true) - 20*np.log10(mag_hat))**2)
    # Phase error (wrap to [-pi,pi])
    ph_true = np.angle(H_true); ph_hat = np.angle(H_hat_i)
    dphi = np.unwrap(ph_true) - np.unwrap(ph_hat)
    # Normalize phase error by pi to keep scale moderate
    ph_err = np.mean((dphi/np.pi)**2)
    return {"bode_mag_mse": float(mag_err), "bode_phase_mse": float(ph_err)}

# ----------------------------------------------------------------------------------
# Largest Lyapunov exponent (Rosenstein method)
# ----------------------------------------------------------------------------------
def lyapunov_rosenstein(t: np.ndarray, y: np.ndarray, m: int=6, tau_steps: int=2, theiler: Optional[int]=None) -> Dict:
    """Estimate the largest Lyapunov exponent from a (possibly multivariate) time series using Rosenstein's method.
    Parameters
    ----------
    t : time grid
    y : array [N, d] or [N,]; if multivariate, use first column
    m : embedding dimension
    tau_steps : delay in samples for embedding
    theiler : Theiler window (exclude temporal neighbors); default = 10*tau_steps
    Returns dict with keys: {'lambda': float, 'fit_range': (i0, i1), 'dt': float}
    """
    y = np.asarray(y)
    if y.ndim == 2:
        y = y[:,0]  # use first coordinate
    N = len(y)
    dt = _uniform_dt(t)
    if theiler is None:
        theiler = 10 * tau_steps
    # Build delay embedding
    M = N - (m-1)*tau_steps
    if M <= 20:
        return {"lambda": np.nan, "fit_range": (0,0), "dt": dt}
    X = np.zeros((M, m))
    for i in range(m):
        X[:,i] = y[i*tau_steps : i*tau_steps + M]
    # For each point, find nearest neighbor beyond Theiler window
    nn_idx = np.zeros(M, dtype=int)
    for i in range(M):
        # Compute distances to all others
        d = np.linalg.norm(X - X[i], axis=1)
        # Exclude self and neighbors within Theiler window
        lo = max(0, i - theiler); hi = min(M, i + theiler + 1)
        d[lo:hi] = np.inf
        j = np.argmin(d)
        nn_idx[i] = j
    # Average log divergence over time
    kmax = min(100, M)  # max horizon to average
    valid = []
    lcurve = np.zeros(kmax)
    counts = np.zeros(kmax)
    for i in range(M):
        j = nn_idx[i]
        k_stop = min(kmax, M - max(i, j))
        if k_stop <= 1: 
            continue
        for k in range(k_stop):
            lcurve[k] += np.log(np.linalg.norm(X[i+k] - X[j+k]) + 1e-12)
            counts[k] += 1
        valid.append(1)
    if np.sum(counts > 0) < 5:
        return {"lambda": np.nan, "fit_range": (0,0), "dt": dt}
    lcurve = lcurve / np.maximum(counts, 1.0)
    # Fit a line on an early linear region (heuristic: first 10..40 samples)
    i0, i1 = 5, min(40, kmax-1)
    x = np.arange(i0, i1) * dt
    yfit = lcurve[i0:i1]
    A = np.vstack([x, np.ones_like(x)]).T
    w, _, _, _ = np.linalg.lstsq(A, yfit, rcond=None)
    lam = float(w[0])  # slope
    return {"lambda": lam, "fit_range": (int(i0), int(i1)), "dt": dt}

# ----------------------------------------------------------------------------------
# Events: zero-crossing detection & timing error
# ----------------------------------------------------------------------------------
def detect_events_zero_crossing(t: np.ndarray, s: np.ndarray, level: float=0.0, direction: str="down") -> np.ndarray:
    """Detect times when s crosses a given level with specified direction.
    direction in {'down','up','both'}.
    Linear interpolation is used between samples to estimate crossing time.
    """
    s = np.asarray(s).reshape(-1)
    times = []
    for k in range(1, len(s)):
        if direction in ("down","both"):
            if s[k-1] > level and s[k] <= level:
                # linear interpolation between (t[k-1], s[k-1]) and (t[k], s[k])
                w = (level - s[k-1]) / (s[k] - s[k-1] + 1e-12)
                times.append(t[k-1] + w*(t[k]-t[k-1]))
        if direction in ("up","both"):
            if s[k-1] < level and s[k] >= level:
                w = (level - s[k-1]) / (s[k] - s[k-1] + 1e-12)
                times.append(t[k-1] + w*(t[k]-t[k-1]))
    return np.array(times)

def event_timing_error(t_true: np.ndarray, s_true: np.ndarray, t_hat: np.ndarray, s_hat: np.ndarray, level: float=0.0, direction: str="down") -> Dict:
    """Compute event timing error (MAE and 95th percentile) between true and predicted signals."""
    T_true = detect_events_zero_crossing(t_true, s_true, level=level, direction=direction)
    T_hat  = detect_events_zero_crossing(t_hat,  s_hat,  level=level, direction=direction)
    if len(T_true) == 0 or len(T_hat) == 0:
        return {"event_mae": float("nan"), "event_p95": float("nan"), "n_true": int(len(T_true)), "n_hat": int(len(T_hat))}
    # Greedy nearest-neighbor pairing
    used = np.zeros(len(T_hat), dtype=bool)
    diffs = []
    for tt in T_true:
        idx = np.argmin(np.abs(T_hat - tt))
        diffs.append(abs(T_hat[idx] - tt))
        used[idx] = True
    diffs = np.array(diffs)
    return {"event_mae": float(np.mean(diffs)), "event_p95": float(np.percentile(diffs, 95)),
            "n_true": int(len(T_true)), "n_hat": int(len(T_hat))}
