# INSTRUCTIONS.md（コーディングエージェント向け指示手順書）

> 本書は、既存成果物（`dynid_benchmark_template_with_edmd_pi.zip` ほか）が**すでにリポジトリに展開済み**である前提で、エージェントが正確に開発・保守できるようにするための詳細ガイドです。

---

## 1. 研究目標と本プログラムの目的

### 研究目標

* **広いサンプリング間隔**や**ノイズ**、**欠測**がある条件でも、データ駆動で**連続時間の力学系**（必要に応じて離散時間モデルで近似）を精度よく同定できることを示す。
* 代表的な **ODE / SDE / PDE / ハイブリッド系（イベント含む）** 上で新手法の**優位性**を、**再現可能な数値実験**により実証する。

### プログラムの目的

1. **真値生成**（A1〜D2の8系）→ **観測化**（粗サンプリング、ジッタ、欠測、外れ値、測定雑音）
2. **学習**（SINDy-STLSQ / EDMD/EDMDc / SINDy-PI / implicit-SINDy / 任意の新手法）→ **ロールアウト評価**
3. **色覚多様性に配慮**した図表出力と**メトリクスの JSON 保存**
4. 将来的な**新手法の追加**・**評価指標の拡張**が容易な API/レイアウト

---

## 2. タスクの大粒度分割（優先順）

1. **レイアウト確認 & 依存導入**

   * `src/dynid_benchmark_template/` 以下にパッケージと `exp/*.yaml` があることを確認
   * `pip install -r requirements.txt`（最小依存：`numpy, matplotlib, pyyaml`）
2. **スモークテスト**

   * `pytest -q` による import と最小実行の確認（短時間設定）
3. **データ生成・実行動作確認**

   * 例：`A1_kappa_sweep.yaml` を SINDy/EDMD/PI/implicit で一通り回す
     `python -m dynid_benchmark.runners.run_experiment --config exp/A1_kappa_sweep.yaml --models sindy_stlsq,edmd,sindy_pi,sindy_implicit,zero --outdir runs`
4. **モデル別 TODO 消化（下記 §3）**

   * STLSQ の平滑オプション, EDMD 条件数監視, PI の可変ステップ重み, implicit の頑健化 など
5. **評価指標/図の拡張**

   * PSD/エネルギー漂い/イベント時刻誤差/FRF など（`evaluation/` と `io/viz.py`）
6. **回帰試験（tests/）の充実**

   * 小データでの `fit→rollout` 健全性、指標の数値窓（寛容度）設定
7. **CI（任意）**

   * GitHub Actions: 最小ケース（A1/B1 短尺 1 seed）でパスすれば OK

---

## 3. モデル別アイディア・数式・TODO

> すべて `dynid_benchmark/models/` 内で `Model` を継承し、`@register_model` でレジストリ登録済み。
> 共通：辞書関数は **多項式**（＋任意で `sin`, `cos`）を既定実装。拡張はライブラリ生成関数を差し替え/引数追加。

### 3.1 SINDy（STLSQ）`sindy_stlsq.py`

**目的**：
[
\dot{X} \approx \Theta(X),\Xi
]

* `Θ`：多項式（＋`sin/cos` 任意）。
* 推定：**逐次しきい値付き最小二乗**（STLSQ）。

**既実装**

* 差分勾配（中心差分＋端点片側差分）、`lam`（しきい値）、`max_iter`
* オプション平滑化（移動平均）。Savitzky–Golay 等も将来差替え可。

**TODO**

* [ ] Savitzky–Golay / Total Variation の平滑オプション（切替可能に）
* [ ] ライブラリ次数・`sin/cos` 有無の**小規模グリッド探索** CLI
* [ ] 係数スパース性（L0/L1 比率）を `metrics` に追加

---

### 3.2 EDMD / EDMDc（離散）`edmd.py`

**離散リフト**：
[
z_{k+1} = A z_k ;(+; B u_k),\quad x_k \approx C z_k, \quad z_k=\Phi(x_k)
]

* **学習**：最小二乗（リッジ正則化あり）
* **条件**：**等間隔サンプリング必須**（runner がチェック）
* **推論**：`predict_next` 実装。`Model.rollout` は離散モデルを自動検知。

**TODO**

* [ ] 辞書の**正規化**と**条件数モニタ**（ログ出力）
* [ ] 連続時間生成子近似（(\logm) で (A\approx e^{G\Delta t}) → (G\approx \frac{1}{\Delta t}\log(A))）※ SciPy 依存のため将来オプション
* [ ] C1-1（PRBS 学習→SINE/Chirp 汎化）向けに **FRF/Bode** の評価ユーティリティ

---

### 3.3 SINDy-PI（積分形式）`sindy_pi.py`

**台形積分での回帰**：
[
\Delta x_k \approx \left(\frac{\Delta t_k}{2}\right)\left[\Theta(x_k)+\Theta(x_{k+1})\right]\Xi
]

* **強み**：数値微分不要 → ノイズに比較的強い
* **学習**：`Δx` と積分設計行列で STLSQ（任意のしきい値）

**TODO**

* [ ] 可変 (\Delta t_k) での**重み付け回帰**（大きなステップの影響制御）
* [ ] **積分ウィンドウ**（複ステップ台形/Simpson）オプションと比較

---

### 3.4 implicit-SINDy（SVD 簡易版）`sindy_implicit.py`

**暗黙回帰**：各次元 (j) について
[
\begin{bmatrix} \dot x_j & \Theta(x) \end{bmatrix} c \approx 0
;\Rightarrow;
\dot x_j \approx \Theta(x),\beta
]

* **実装**：最小特異ベクトルから (\alpha, \beta) を抽出。微分は中心差分。
* **位置付け**：粗サンプリングで脆弱（基準ベースライン）。

**TODO**

* [ ] **RANSAC** もしくは **SVHT** を用いた SVD の頑健化
* [ ] 微分ノイズ対策（Savitzky–Golay, TV フィルタ）
* [ ] 退化ケース（(\alpha\simeq 0)）の**一貫したフォールバック**整理

---

## 4. 系（データ生成）ごとの要点

* **A1 乾燥摩擦**（`tanh(κv)` 平滑化）

  * 目的：**非滑らか起因の誤差**に対する頑健性検証（κ スイープ）
* **A2 バウンシングボール**（hard/soft）

  * 目的：**イベント**（衝突）を含むハイブリッド系での外挿健全性
* **B1 OU**（線形安定）

  * 目的：SNR スイープでの**ノイズ耐性**
* **B2 二井戸 SDE**

  * 目的：長時間での**二井戸遷移**（統計的一致性）
* **C1-1 質量ばね＋入力**

  * 目的：**入力一般化**（PRBS 学習→SINE/Chirp 評価）
* **C1-2 強制 Duffing**

  * 目的：**帯域外外挿**の頑健性
* **D1 Burgers（1D）/ D2 KS**

  * 目的：**PDE 半離散 ODE** 同定；**スペクトル/統計**比較（将来指標）

---

## 5. 可視化とアクセシビリティ

* 図は**色に依存せず**、**線種（—, --, -., :）×マーカー（○/□/△/◇/×/+ …）**で**自動切替**（`io/viz.py`）
* グリッドと凡例を既定 ON。**1 図 1 チャート**を基本。
* 追加する図も同方針に合わせること。

---

## 6. 実行コマンド（代表例）

```bash
# A1 乾燥摩擦：複数モデル比較
python -m dynid_benchmark.runners.run_experiment \
  --config exp/A1_kappa_sweep.yaml \
  --models sindy_stlsq,edmd,sindy_pi,sindy_implicit,zero \
  --outdir runs
```

* 生成物：`runs/<exp_id>/<tag>/`

  * `data_train.npz`, `data_test.npz`, `metrics_*.json`, `rollout_*.png`
  * 失敗時は `error_<model>.txt` に理由を出力

---

## 7. テストと品質ゲート

* `tests/test_smoke.py`：主要モジュール import・短尺実行のスモーク
* `tests/test_models.py`：小データで `fit→rollout` が**例外なく**走るか
* 将来：数値窓（RMSE < しきい値）でモデル回帰

---

## 8. コーディング規約（要点）

* 依存は最小限（`numpy, matplotlib, pyyaml`）。外部最適化/SciPy は**任意/将来**。
* 新手法は `models/base.py: Model` を継承、`@register_model` 必須。
* 例外は**握りつぶさず** runner が `error_*.txt` を吐く設計を維持。
* 数式に関わる変更は本書の当該セクションを更新すること。

---

# AGENTS.md（実行・テストの超要約）

## 1) セットアップ

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 最小実行（例：A1 × 各モデル）

```bash
python -m dynid_benchmark.runners.run_experiment \
  --config exp/A1_kappa_sweep.yaml \
  --models sindy_stlsq,edmd,sindy_pi,sindy_implicit,zero \
  --outdir runs
```

* 出力先：`runs/<exp>/<tag>/`

  * 図：`rollout_<model>.png`
  * 指標：`metrics_<model>.json`
  * エラー：`error_<model>.txt`（等間隔でない EDMD など）

## 3) スモークテスト

```bash
pip install pytest
pytest -q
```

## 4) よくある注意

* **EDMD/EDMDc は等間隔サンプリング必須**（`sampling.jitter_pct`/`missing_pct` を 0 に）
* **SINDy/STLSQ と implicit-SINDy は数値微分**を使うため、粗サンプリングで不利：

  * `smooth_window` や将来の Savitzky–Golay を試す
* **SINDy-PI は積分**で回帰するため、ノイズに比較的強いが**大ステップ**では重み付けが有効

## 5) 新手法の追加

* `dynid_benchmark/models/your_method.py` を作成し `Model` を継承、`@register_model` を付与
* `predict_derivative`（連続時間）or `predict_next`（離散時間）のどちらかを実装
* 実行時に `--models your_method` を指定

## 6) アクセシビリティ

* 図は**線種×マーカー**で識別（色指定なし）。`io/viz.py` のスタイルに従うこと。

