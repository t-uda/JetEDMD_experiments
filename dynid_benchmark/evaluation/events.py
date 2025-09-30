import numpy as np


def find_level_crossings(t, y, level=0.0):
    """
    信号 y が指定レベル（デフォルト 0.0）を横切る時刻を線形補間で推定する。

    Parameters
    ----------
    t : ndarray (N,)
        単調増加と仮定する時刻列。
    y : ndarray (N,) or (N, d)
        観測信号。多次元の場合は y[:, 0] を判定に用いる。
    level : float
        交差判定するしきい値。

    Returns
    -------
    t_events : ndarray (K,)
        線形補間によりサブサンプル精度で推定された交差時刻。
    """
    yy = y[:, 0] if y.ndim == 2 else y
    s = yy - level
    sign = np.sign(s)
    # 隣接サンプルで符号が変化している箇所を交差候補とみなす
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    t_events = []
    for i in idx:
        # 連続する 2 サンプル間を線形補間して交差時刻を推定
        t0, t1 = t[i], t[i + 1]
        y0, y1 = yy[i], yy[i + 1]
        if y1 == y0:
            tau = 0.5
        else:
            tau = (level - y0) / (y1 - y0)
        tau = np.clip(tau, 0.0, 1.0)
        t_events.append(t0 + tau * (t1 - t0))
    return np.asarray(t_events, dtype=float)


def match_events(t_true, t_pred, tol=None):
    """
    予測イベントと真値イベントを貪欲法で近傍マッチングする。

    Parameters
    ----------
    t_true : ndarray (K,)
        真値イベント時刻。
    t_pred : ndarray (M,)
        予測イベント時刻。
    tol : float or None
        None 以外なら |Δt| > tol の組み合わせを破棄する。

    Returns
    -------
    pairs : list of (i_true, i_pred, dt)
        マッチした添字ペアと signed 誤差 dt = t_pred - t_true。
    """
    if len(t_true) == 0 or len(t_pred) == 0:
        return []
    used = np.zeros(len(t_pred), dtype=bool)
    pairs = []
    for i, tt in enumerate(t_true):
        j = np.argmin(np.abs(t_pred - tt))
        if used[j]:
            continue
        dt = float(t_pred[j] - tt)
        if (tol is None) or (abs(dt) <= tol):
            pairs.append((i, j, dt))
            used[j] = True
    return pairs


def event_timing_metrics(t_true, t_pred, tol=None):
    """
    イベント時刻誤差の要約統計量を計算する。

    Returns
    -------
    stats : dict
        {'n_true':..., 'n_pred':..., 'n_matched':..., 'mae':..., 'median':..., 'p95':..., 'bias':...}
    """
    pairs = match_events(t_true, t_pred, tol=tol)
    if not pairs:
        return {
            "n_true": len(t_true),
            "n_pred": len(t_pred),
            "n_matched": 0,
            "mae": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "bias": np.nan,
        }
    dts = np.array([dt for _, _, dt in pairs], dtype=float)
    return {
        "n_true": int(len(t_true)),
        "n_pred": int(len(t_pred)),
        "n_matched": int(len(pairs)),
        "mae": float(np.mean(np.abs(dts))),
        "median": float(np.median(dts)),
        "p95": float(np.percentile(np.abs(dts), 95)),
        "bias": float(np.mean(dts)),
    }
