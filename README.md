

## 追加の評価ユーティリティ（論文図向け）

### FRF / Bode（Welch 推定）
```python
from dynid_benchmark.evaluation import frf_welch, bode_mag_phase
fs = 1.0 / np.mean(np.diff(t_train))
f, H, coh, Syu, Suu = frf_welch(u_train[:,0], y_train[:,0], fs=fs, nperseg=1024)
mag_db, phase_deg = bode_mag_phase(H, deg=True)
```

### PSD（Welch）
```python
from dynid_benchmark.evaluation import psd_welch
fs = 1.0 / np.mean(np.diff(t_test))
f, Pxx = psd_welch(y_test[:,0], fs=fs, nperseg=1024)
```

### 最大リアプノフ指数（Rosenstein）
```python
from dynid_benchmark.evaluation import rosenstein_lmax
fs = 1.0 / np.mean(np.diff(t_test))
lmax, j_axis, y_log = rosenstein_lmax(y_test[:,0], fs=fs, m=6, tau=None, theiler=10, fit_range=(5,50))
```

### イベント時刻誤差（バウンシングボール等）
```python
from dynid_benchmark.evaluation import find_level_crossings, event_timing_metrics
t_true_evt = find_level_crossings(t_true, y_true[:,0], level=0.0)
t_pred_evt = find_level_crossings(t_pred, y_pred[:,0], level=0.0)
stats = event_timing_metrics(t_true_evt, t_pred_evt, tol=0.05)  # 50ms 許容
```

> **注**：上記は数値例です。論文では、同一の `fs`・窓長・重複率等を**全手法で揃える**と公平な比較になります。

