# SINDy 関連メモ（任意利用）

> このファイルは SINDy 系モデルを利用する際の補足情報をまとめています。標準運用は PyKoopman を想定しているため、SINDy を使う場合のみ参照してください。

## 環境と依存

- **Poetry グループ `pysindy`** を有効化すると、`pysindy` 1.7 系と周辺依存を追加できます。
  ```bash
  poetry install --with pysindy
  ```
- **SINDy-PI** など `cvxpy` に依存する機能は SciPy ≥1.13 を要求し、PyKoopman（SciPy ≤1.11.2 依存）とは同一環境で共存しません。必要に応じて別仮想環境（例：`envs/pysindy`）を利用してください。
- NumPy 2.0 系を利用する場合は、`pysindy` の互換性パッチ（`np.math = math`）が有効であるかを都度確認します。

## モデル実装メモ

### SINDy（STLSQ）`dynid_benchmark/models/sindy_stlsq.py`

- **目的**：逐次しきい値付き最小二乗（STLSQ）で連続時間システムの係数を推定。
- **辞書関数**：多項式基底（任意で `sin` / `cos` 追加）。
- **既存オプション**
  - 差分勾配（中心差分＋端点片側差分）。
  - `lam`（しきい値）、`max_iter` の調整パラメータ。
  - 移動平均による簡易平滑化。
- **TODO**
  - Savitzky–Golay や Total Variation など切替可能な平滑オプション。
  - 辞書次数・三角関数有無の小規模グリッド探索 CLI。
  - 係数スパース性（L0/L1 比率）のメトリクス化。

### SINDy-PI（積分形式）`dynid_benchmark/models/sindy_pi.py`

- **回帰式**：
  \[
  \Delta x_k \approx \left(\frac{\Delta t_k}{2}\right)\left[\Theta(x_k) + \Theta(x_{k+1})\right] \Xi
  \]
- **長所**：数値微分が不要でノイズに比較的強い。
- **課題 / TODO**
  - 可変 \(\Delta t_k\) に対する重み付き回帰。
  - 複数ステップ積分（台形 / Simpson）オプションの比較。
- **PySINDy-PI**：`pysindy` グループを有効化すると `pysindy_pi` モデルが利用できます。`cvxpy` を別途追加し、SciPy ≥1.13 の環境で実行してください。

### implicit-SINDy `dynid_benchmark/models/sindy_implicit.py`

- **暗黙回帰**：各次元 \(j\) について
  \[
  \begin{bmatrix} \dot x_j & \Theta(x) \end{bmatrix} c \approx 0 \quad\Rightarrow\quad \dot x_j \approx \Theta(x) \beta
  \]
- **実装**：最小特異ベクトルから係数を抽出。数値微分は中心差分を使用。
- **TODO**
  - RANSAC や Singular Value Hard Thresholding による頑健化。
  - 微分ノイズ対策（Savitzky–Golay / TV フィルタ）。
  - 係数退化（\(\alpha \simeq 0\)）時のフォールバック整理。

## 実験運用の注意

1. **ノイズ条件**：SINDy 系モデルは数値微分に敏感なため、初期検証はノイズ無し（`SNR_dB = null`）で行い、動作確認後に高 SNR → 低 SNR の順で評価します。
2. **YAML 設定**：`exp/` のモデルリストに `sindy_*` を追加する際は、辞書次数・平滑化窓・最適化ハイパを明示し、再現性を確保します。
3. **ログ**：失敗時は `runs/<exp>/<tag>/error_sindy_*.txt` を確認し、微分設定や正則化を調整します。
4. **比較**：PyKoopman との比較結果は `docs/model_comparison.md` などに追記し、SINDy を使わない既定フローとの差異を明確にします。

## 参考リンク

- PySINDy ドキュメント: <https://pysindy.readthedocs.io/>
- SINDy-PI チュートリアル: <https://pysindy.readthedocs.io/en/stable/examples/9_sindypi_with_sympy/example.html>
- implicit-SINDy 解説: <https://pysindy.readthedocs.io/en/latest/examples/index.html>

> SINDy を利用する場合も、成果物は `runs/<exp>/<tag>/` に統一されます。大容量ファイルの扱いや整形・テストの運用ルールは PyKoopman 系と同じです。
