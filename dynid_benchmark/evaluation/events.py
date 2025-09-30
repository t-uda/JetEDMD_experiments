import numpy as np

def find_level_crossings(t, y, level=0.0):
    """
    Detect times where y crosses a given level (default 0.0) using linear interpolation.

    Parameters
    ----------
    t : ndarray (N,)
        Time stamps (assumed increasing).
    y : ndarray (N,) or (N, d)
        Signal; for multi-dim, the first column y[:,0] is used.
    level : float
        Crossing level.

    Returns
    -------
    t_events : ndarray (K,)
        Estimated event times (sub-sample accuracy by linear interpolation).
    """
    yy = y[:,0] if y.ndim == 2 else y
    s = yy - level
    sign = np.sign(s)
    # sign changes between consecutive samples indicate a crossing
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    t_events = []
    for i in idx:
        # Linear interpolation between (t[i], y[i]) and (t[i+1], y[i+1])
        t0, t1 = t[i], t[i+1]
        y0, y1 = yy[i], yy[i+1]
        if y1 == y0:
            tau = 0.5
        else:
            tau = (level - y0) / (y1 - y0)
        tau = np.clip(tau, 0.0, 1.0)
        t_events.append(t0 + tau * (t1 - t0))
    return np.asarray(t_events, dtype=float)

def match_events(t_true, t_pred, tol=None):
    """
    Greedy nearest-neighbor matching of predicted events to true events.

    Parameters
    ----------
    t_true : ndarray (K,)
    t_pred : ndarray (M,)
    tol : float or None
        If not None, drop pairs with |Δt| > tol.

    Returns
    -------
    pairs : list of (i_true, i_pred, dt)
        Matches with signed time error dt = t_pred - t_true.
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
    Compute summary metrics for event timing errors.

    Returns
    -------
    stats : dict
        {'n_true':..., 'n_pred':..., 'n_matched':..., 'mae':..., 'median':..., 'p95':..., 'bias':...}
    """
    pairs = match_events(t_true, t_pred, tol=tol)
    if not pairs:
        return {'n_true': len(t_true), 'n_pred': len(t_pred), 'n_matched': 0,
                'mae': np.nan, 'median': np.nan, 'p95': np.nan, 'bias': np.nan}
    dts = np.array([dt for _,_,dt in pairs], dtype=float)
    return {
        'n_true': int(len(t_true)),
        'n_pred': int(len(t_pred)),
        'n_matched': int(len(pairs)),
        'mae': float(np.mean(np.abs(dts))),
        'median': float(np.median(dts)),
        'p95': float(np.percentile(np.abs(dts), 95)),
        'bias': float(np.mean(dts)),
    }
