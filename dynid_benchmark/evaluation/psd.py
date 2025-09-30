import numpy as np
from scipy import signal

def psd_welch(x, fs, nperseg=None, detrend='constant', window='hann', noverlap=None):
    """
    Power Spectral Density (Welch).

    Parameters
    ----------
    x : ndarray, shape (N,) or (N, d)
        Time series (possibly multi-channel). Each channel is processed independently.
    fs : float
        Sampling frequency [Hz].
    Other params are passed to scipy.signal.welch.

    Returns
    -------
    f : ndarray (F,)
        Frequencies (Hz).
    Pxx : ndarray (F,) or (F, d)
        PSD estimates per channel.
    """
    x = np.atleast_2d(x)
    if x.shape[0] < x.shape[1]:  # ensure shape (N, d)
        x = x.T
    P_list = []
    F = None
    for j in range(x.shape[1]):
        f, P = signal.welch(x[:,j], fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend, window=window)
        if F is None: F = f
        P_list.append(P)
    Pxx = np.stack(P_list, axis=1) if len(P_list)>1 else P
    return F, Pxx.squeeze()
