import numpy as np
from scipy import signal


def frf_welch(u, y, fs, nperseg=None, detrend="constant", window="hann", noverlap=None):
    """
    Estimate Frequency Response Function (FRF) H(f) = S_yu(f) / S_uu(f) via Welch's method.

    Parameters
    ----------
    u : ndarray, shape (N,) or (N, m)
        Input time series (single or multi-input). If multi-input, this returns
        FRF for each input independently (no MIMO cross-talk removal).
    y : ndarray, shape (N,) or (N, p)
        Output time series.
    fs : float
        Sampling frequency [Hz] (fs = 1/Δt).
    nperseg, detrend, window, noverlap : passed to scipy.signal.csd/welch

    Returns
    -------
    f : ndarray
        Frequency vector (Hz).
    H : ndarray
        Complex FRF array. If y is (N,) and u is (N,), shape is (F,).
        If y is (N,p) and u is (N,m), shape is (F, p, m).
    coh : ndarray
        Magnitude-squared coherence γ^2(f) between u and y (same shape as |H|).
    Syu, Suu : ndarray
        Cross/auto power spectra (for diagnostics).
    """
    u = np.atleast_2d(u)  # (m, N) if originally (N,)
    if u.shape[0] < u.shape[1]:  # heuristic: ensure shape (N, m)
        u = u.T
    y = np.atleast_2d(y)
    if y.shape[0] < y.shape[1]:
        y = y.T
    N = u.shape[0]
    F = None
    H_list = []
    C_list = []
    Syu_list = []
    Suu_list = []
    for pi in range(y.shape[1]):
        H_cols = []
        C_cols = []
        Syu_cols = []
        Suu_cols = []
        for mi in range(u.shape[1]):
            f, Puy = signal.csd(
                y[:, pi],
                u[:, mi],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=detrend,
                window=window,
            )
            _, Puu = signal.welch(
                u[:, mi],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=detrend,
                window=window,
            )
            _, Coh = signal.coherence(
                y[:, pi],
                u[:, mi],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=detrend,
                window=window,
            )
            H_cols.append(Puy / (Puu + 1e-15))
            C_cols.append(Coh)
            Syu_cols.append(Puy)
            Suu_cols.append(Puu)
            if F is None:
                F = f
        H_list.append(np.stack(H_cols, axis=1))  # (F, m)
        C_list.append(np.stack(C_cols, axis=1))  # (F, m)
        Syu_list.append(np.stack(Syu_cols, axis=1))
        Suu_list.append(np.stack(Suu_cols, axis=1))
    H = np.stack(H_list, axis=1)  # (F, p, m)
    coh = np.stack(C_list, axis=1)  # (F, p, m)
    Syu = np.stack(Syu_list, axis=1)
    Suu = np.stack(Suu_list, axis=1)
    return F, H.squeeze(), coh.squeeze(), Syu.squeeze(), Suu.squeeze()


def bode_mag_phase(H, deg=True):
    """
    Convert complex FRF to magnitude [dB] and phase [deg or rad].
    H : ndarray (...,)
    """
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-15))
    ang = np.angle(H, deg=deg)
    return mag_db, ang
