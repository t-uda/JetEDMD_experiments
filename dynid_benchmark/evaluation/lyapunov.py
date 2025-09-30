import numpy as np

def _embed(x, m, tau):
    """
    Delay-coordinate embedding: R^N -> R^{N-(m-1)tau} by stacking delayed copies.

    Returns
    -------
    X_emb : ndarray, shape (M, m)
        Embedded trajectory, M = N - (m-1)tau
    """
    N = len(x)
    M = N - (m-1)*tau
    if M <= 1:
        raise ValueError("Time series too short for the requested embedding.")
    X = np.zeros((M, m), dtype=float)
    for i in range(m):
        X[:, i] = x[i*tau : i*tau + M]
    return X

def rosenstein_lmax(x, fs, m=6, tau=None, theiler=10, fit_range=(5, 50)):
    """
    Rosenstein's method to estimate the largest Lyapunov exponent (L_max) from a scalar time series.
    Ref: Rosenstein et al., Physica D 65 (1993) 117–134.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Scalar time series (use one state component, or a principal component).
    fs : float
        Sampling frequency [Hz].
    m : int
        Embedding dimension.
    tau : int or None
        Delay (in samples). If None, pick the smallest lag where autocorr < 1/e.
    theiler : int
        Theiler window (temporal decorrelation window) to avoid trivial neighbors.
    fit_range : tuple (j_start, j_end)
        Range of forward steps (in samples) over which we fit the slope of <log distance>.

    Returns
    -------
    lmax : float
        Estimated largest Lyapunov exponent [1/s].
    j_axis : ndarray
        Indices used for the regression (samples).
    y_mean : ndarray
        Mean log-separation curve for diagnostics.
    """
    x = np.asarray(x, dtype=float).ravel()

    # Pick tau by autocorrelation crossing 1/e (if not provided)
    if tau is None:
        ac = np.correlate(x - x.mean(), x - x.mean(), mode='full')
        ac = ac[ac.size//2:]
        ac = ac / ac[0]
        tau = int(np.argmax(ac < 1/np.e))
        tau = max(tau, 1)

    # Build embedding
    X = _embed(x, m=m, tau=tau)   # (M, m)
    M = X.shape[0]

    # For each point, find nearest neighbor with Theiler exclusion
    from numpy.linalg import norm
    dists = np.full(M, np.inf)
    idx_nn = np.full(M, -1, dtype=int)
    for i in range(M):
        # Compute distances to all points except within theiler window
        j = np.arange(M)
        mask = (np.abs(j - i) > theiler)
        if np.any(mask):
            d = norm(X[mask] - X[i], axis=1)
            k = np.argmin(d)
            dists[i] = d[k]
            idx_nn[i] = j[mask][k]

    # Compute mean log separation over forward times j
    j_start, j_end = fit_range
    j_axis = np.arange(j_start, j_end)
    y_log = []
    for j in j_axis:
        # valid indices where i+j and nn_i + j are within range
        mask = (np.arange(M) + j < M) & (idx_nn >= 0) & (idx_nn + j < M)
        if not np.any(mask):
            y_log.append(np.nan)
            continue
        sep = np.linalg.norm(X[mask][:,None,:] - X[idx_nn[mask]+j][:,None,:], axis=2).squeeze()
        # Avoid zeros
        sep = np.maximum(sep, 1e-15)
        y_log.append(np.mean(np.log(sep)))
    y_log = np.asarray(y_log)

    # Linear regression y = a + b * (j / fs)
    mask = np.isfinite(y_log)
    if np.sum(mask) < 3:
        return np.nan, j_axis, y_log
    t_axis = j_axis[mask] / fs
    y = y_log[mask]
    A = np.vstack([np.ones_like(t_axis), t_axis]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coef[1]   # slope in log distance per second
    return float(slope), j_axis, y_log
