# INSTRUCTIONS.md（コーディングエージェント向け指示手順書）

> 本書は、`dynid_benchmark` を基盤とした PyKoopman 中心の実験テンプレートを正しく保守・拡張するためのガイドです。SINDy 系メモは `docs/sindy_notes.md` に分離しました。

---

## 1. 研究目標と本プログラムの目的

### 研究目標

* **粗いサンプリング**や**ノイズ／欠測**といった厳しい観測条件下でも、データ駆動で連続時間力学系（必要に応じて離散近似）を精度良く推定できることを示す。
* 代表的な **ODE / SDE / PDE / ハイブリッド系（イベント含む）** 上で、Koopman 系アプローチの優位性と限界を再現可能な数値実験で示す。

### プログラムの目的

1. **真値生成**（A1〜D2 などの系列）→ **観測化**（粗サンプリング、ジッタ、欠測、外れ値、測定雑音）
2. **学習**（PyKoopman-EDMD / PyKoopman-EDMDc / 教育実装 EDMD / Zero / MeanDerivative 等）→ **ロールアウト評価**
3. **色覚多様性に配慮**した図表と **メトリクス JSON** の自動生成
4. 将来的な**新手法の追加**・**評価指標の拡張**が容易な API/レイアウトの維持

---

## 2. タスクの大粒度分割（優先順）

1. **レイアウト確認 & 依存導入**
   - `dynid_benchmark/` と `exp/*.yaml` が揃っていることを確認。
   - `poetry install` で標準依存（PyKoopman スタック）を導入。追加依存は `docs/NEW_LIBRARIES.md` を参照。
2. **スモークテスト**
   - `poetry run pytest -q` で import と短尺実行を確認。
   - **ノイズ無し（`SNR_dB: [null, ...]`）** を初期検証の既定とし、成功後にノイズケースへ展開。
3. **PyKoopman 実験動作確認**
   - 例：`A1_kappa_sweep.yaml` を `pykoopman_edmd,zero` で実行し、出力ファイルと `error_pykoopman_*.txt` を確認。
   - 制御入力ありのケースでは `pykoopman_edmdc` を利用し、`exp/` 側の入力長と一致させる。
4. **モデル別 TODO 消化（PyKoopman/EDMD 系）**
   - 辞書正規化・条件数モニタリングなど、Koopman 系の安定化タスクを優先。
5. **評価指標/図の拡張**
   - PSD / エネルギー漂い / イベント時刻誤差 / FRF などを `evaluation/` と `io/viz.py` で追加検討。
6. **回帰試験（tests/）の強化**
   - `tests/test_models_pykoopman.py` 等をベースに長期ロールアウトやノイズ混入ケースを拡充。
7. **CI（任意）**
   - GitHub Actions で A1/B1 の短尺ケースを実行し、再現性を担保。

> SINDy 系の運用タスクは `docs/sindy_notes.md` に隔離しました。必要時のみ参照してください。

---

## 3. PyKoopman / EDMD モデル運用メモ

> すべて `dynid_benchmark/models` 配下で `Model` を継承し、`@register_model` でレジストリ登録されています。

### 3.1 PyKoopman-EDMD / PyKoopman-EDMDc

* **概要**：`pykoopman.Koopman` をラップし、多項式観測（`Polynomial`）を用いた EDMD / EDMDc を提供。
* **前提条件**：等間隔サンプリング。`sampling.jitter_pct` や `missing_pct` を使うケースでは、リサンプリングや条件分岐を事前に検討する。
* **エラーハンドリング**：学習失敗時には `runs/<exp>/<tag>/error_pykoopman_*.txt` に詳細を保存。条件数悪化やデータ不足を疑う。
* **今後の検討事項**
  - 辞書行列の正規化と条件数ログ出力
  - 制御入力付きケースでの FRF/Bode 評価ユーティリティ
  - ノイズ混入データでのロバスト評価テンプレート整備

### 3.2 教育実装 EDMD / 離散モデル

* **離散リフト**：
  \[ z_{k+1} = A z_k (+ B u_k),\quad x_k \approx C z_k,\quad z_k = \Phi(x_k) \]
* **学習**：リッジ付き最小二乗。`Model.rollout` で離散モデルを自動検出。
* **運用 Tips**
  - 多項式基底の次数と入力有無を YAML で明示。
  - 条件数が高い場合は基底の縮小や正則化係数の増加を検討。
  - PyKoopman との比較で差異があれば `docs/model_comparison.md` に記録する。

### 3.3 ベースライン（Zero / MeanDerivative など）

* **Zero**：定常ベースライン。故障時のフォールバック確認に利用。
* **MeanDerivative**：平均勾配で一次近似を行う単純モデル。粗いサンプルでの下限指標として扱う。

---

## 4. 系（データ生成）ごとの要点

* **A1 乾燥摩擦**：κ スイープで非滑らか起因の誤差を検証。
* **A2 バウンシングボール**：イベント（衝突）を含むハイブリッド系。イベント時刻の再現性を重視。
* **B1 OU**：SNR スイープでノイズ耐性を評価。
* **B2 二井戸 SDE**：長時間統計（井戸遷移頻度）を確認。
* **C1-1 質量ばね＋入力**：PRBS 学習 → SINE/Chirp 評価で入力一般化を検証。
* **C1-2 強制 Duffing**：帯域外外挿の健全性を確認。
* **D1 Burgers / D2 Kuramoto–Sivashinsky**：PDE 半離散 ODE。スペクトル/エネルギー指標を将来的に追加。

---

## 5. 可視化とアクセシビリティ

* 図は色に依存せず、線種（—, --, -., :）×マーカー（○/□/△/◇/×/+ …）で識別（`dynid_benchmark/io/viz.py`）。
* グリッドと凡例を既定 ON。1 図 1 チャートを基本とし、必要に応じて射影次元を制御。

---

## 6. 実行コマンド（代表例）

```bash
# PyKoopman-EDMD を用いた A1 系列の検証
poetry run python -m dynid_benchmark.runners.run_experiment \
  --config exp/A1_kappa_sweep.yaml \
  --models pykoopman_edmd,zero \
  --outdir runs
```

* 出力物：`runs/<exp_id>/<tag>/`
  - `data_train.npz`, `data_test.npz`
  - `metrics_<model>.json`
  - `rollout_<model>.png`
  - 失敗時は `error_<model>.txt`

---

## 7. テストと品質ゲート

* `tests/test_smoke.py`：主要モジュール import・短尺実行のスモーク。
* `tests/test_models_pykoopman.py`：PyKoopman-EDMD/EDMDc の学習とロールアウトが例外なく走るかを確認。
* 将来タスク：長期ロールアウトやノイズ混入ケースの回帰テストを追加し、指標の許容誤差を調整。

---

## 8. 参考リンク

* PyKoopman：<https://pykoopman.readthedocs.io/>
* PyDMD：<https://pydmd.github.io/PyDMD/>
* SINDy 系の詳細：`docs/sindy_notes.md`
