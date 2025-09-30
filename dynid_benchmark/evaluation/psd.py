import numpy as np
from scipy import signal


def psd_welch(x, fs, nperseg=None, detrend="constant", window="hann", noverlap=None):
    """
    Welch 法によるパワースペクトル密度 (PSD) 推定を行う。

    Parameters
    ----------
    x : ndarray, shape (N,) or (N, d)
        単一もしくは多チャンネルの時系列。各チャンネルを独立に処理する。
    fs : float
        サンプリング周波数 [Hz]。
    その他の引数は scipy.signal.welch に転送される。

    Returns
    -------
    f : ndarray (F,)
        周波数ベクトル [Hz]。
    Pxx : ndarray (F,) or (F, d)
        チャンネルごとの PSD 推定値。
    """
    x = np.atleast_2d(x)
    if x.shape[0] < x.shape[1]:  # ensure shape (N, d)
        x = x.T
    P_list = []
    F = None
    for j in range(x.shape[1]):
        f, P = signal.welch(
            x[:, j],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            window=window,
        )
        if F is None:
            F = f
        P_list.append(P)
    Pxx = np.stack(P_list, axis=1) if len(P_list) > 1 else P
    return F, Pxx.squeeze()
