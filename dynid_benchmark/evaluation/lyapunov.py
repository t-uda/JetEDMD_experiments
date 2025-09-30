import numpy as np


def _embed(x, m, tau):
    """
    遅延座標埋め込み：遅延付きの写像で R^N -> R^{N-(m-1)tau} を構成する。

    Returns
    -------
    X_emb : ndarray, shape (M, m)
        埋め込み軌道。M = N - (m-1)tau。
    """
    N = len(x)
    M = N - (m - 1) * tau
    if M <= 1:
        raise ValueError("Time series too short for the requested embedding.")
    X = np.zeros((M, m), dtype=float)
    for i in range(m):
        X[:, i] = x[i * tau : i * tau + M]
    return X


def rosenstein_lmax(x, fs, m=6, tau=None, theiler=10, fit_range=(5, 50)):
    """
    ローゼンスタイン法で最大リアプノフ指数 (L_max) をスカラー時系列から推定する。
    参考: Rosenstein et al., Physica D 65 (1993) 117–134.

    Parameters
    ----------
    x : ndarray, shape (N,)
        スカラー時系列（単一状態や主成分など）。
    fs : float
        サンプリング周波数 [Hz]。
    m : int
        埋め込み次元。
    tau : int or None
        遅延（サンプル単位）。None の場合は自己相関が 1/e を下回る最小ラグを採用。
    theiler : int
        テイラー窓（近接点の排除幅）。
    fit_range : tuple (j_start, j_end)
        平均対数距離の傾きをフィットするステップ範囲（サンプル単位）。

    Returns
    -------
    lmax : float
        推定された最大リアプノフ指数 [1/s]。
    j_axis : ndarray
        回帰に用いたステップ指標。
    y_mean : ndarray
        診断用の平均対数距離曲線。
    """
    x = np.asarray(x, dtype=float).ravel()

    # 遅延が未指定なら自己相関が 1/e を下回る最小ラグを採用
    if tau is None:
        ac = np.correlate(x - x.mean(), x - x.mean(), mode="full")
        ac = ac[ac.size // 2 :]
        ac = ac / ac[0]
        tau = int(np.argmax(ac < 1 / np.e))
        tau = max(tau, 1)

    # 埋め込み軌道を構築
    X = _embed(x, m=m, tau=tau)  # (M, m)
    M = X.shape[0]

    # 各点に対しテイラー窓内を除外した最短距離の近傍を探索
    from numpy.linalg import norm

    dists = np.full(M, np.inf)
    idx_nn = np.full(M, -1, dtype=int)
    for i in range(M):
        # テイラー窓外の候補点との距離を計算
        j = np.arange(M)
        mask = np.abs(j - i) > theiler
        if np.any(mask):
            d = norm(X[mask] - X[i], axis=1)
            k = np.argmin(d)
            dists[i] = d[k]
            idx_nn[i] = j[mask][k]

    # 前進ステップ j ごとに平均対数距離を算出
    j_start, j_end = fit_range
    j_axis = np.arange(j_start, j_end)
    y_log = []
    for j in j_axis:
        # i+j と最近傍 idx_nn + j が範囲内に収まるもののみ評価
        mask = (np.arange(M) + j < M) & (idx_nn >= 0) & (idx_nn + j < M)
        if not np.any(mask):
            y_log.append(np.nan)
            continue
        sep = np.linalg.norm(
            X[mask][:, None, :] - X[idx_nn[mask] + j][:, None, :], axis=2
        ).squeeze()
        # ゼロ距離を避けるため最小値を制限
        sep = np.maximum(sep, 1e-15)
        y_log.append(np.mean(np.log(sep)))
    y_log = np.asarray(y_log)

    # y = a + b * (j / fs) の線形回帰で傾きを求める
    mask = np.isfinite(y_log)
    if np.sum(mask) < 3:
        return np.nan, j_axis, y_log
    t_axis = j_axis[mask] / fs
    y = y_log[mask]
    A = np.vstack([np.ones_like(t_axis), t_axis]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coef[1]  # slope in log distance per second
    return float(slope), j_axis, y_log
