import numpy as np
from scipy import signal


def frf_welch(u, y, fs, nperseg=None, detrend="constant", window="hann", noverlap=None):
    """
    Welch 法で周波数応答関数 H(f) = S_yu(f) / S_uu(f) を推定する。

    Parameters
    ----------
    u : ndarray, shape (N,) or (N, m)
        入力時系列（単入力または多入力）。多入力の場合、入力間の相互影響は除去しない。
    y : ndarray, shape (N,) or (N, p)
        出力時系列。
    fs : float
        サンプリング周波数 [Hz]（fs = 1/Δt）。
    nperseg, detrend, window, noverlap : scipy.signal.csd / welch へそのまま引き渡す引数。

    Returns
    -------
    f : ndarray
        周波数ベクトル [Hz]。
    H : ndarray
        複素 FRF。y, u が単入力単出力なら (F,), 多入力多出力なら (F, p, m)。
    coh : ndarray
        u と y のコヒーレンス γ^2(f)（|H| と同じ形状）。
    Syu, Suu : ndarray
        交差／自己パワースペクトル（診断用）。
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
            H_cols.append(Puy / (Puu + 1e-15))  # 数値的に安定化するため微小量を加算
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
    複素 FRF から振幅（dB）と位相（deg または rad）を算出する。
    H : ndarray (...,)
    """
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-15))
    ang = np.angle(H, deg=deg)
    return mag_db, ang
