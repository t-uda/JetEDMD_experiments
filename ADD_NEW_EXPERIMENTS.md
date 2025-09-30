# エージェント指示プロンプト（E1–E4 実験追加）

## 1. スコープと到達目標

* 本研究では「サンプル間隔が粗くても力学系の同定が可能」であることを示す。
* 実験対象は以下の **Core-4**：

  1. **E1: Lorenz-63**（カオス、長期統計量）
  2. **E2: 質量–ばね–ダンパ＋入力**（LTI系、未見入力一般化）
  3. **E3: 1D Burgers 方程式（半離散 ODE）**（PDEをODEとして扱う）
  4. **E4: 乾燥摩擦振動子**（非滑らか度のスイープによる適用境界の可視化）
* 比較対象：SINDy-STLSQ, SINDy-PI, implicit-SINDy, EDMD/EDMDc, Zero。
* 成果物：図1a, 1b, 2a, 2b, 3, 4 と 表1（CSV/TEX）。
* 可視化は**色覚多様性に配慮（線種×マーカー、色指定に依存しない）**。

---

## 2. 物理的な問題設定

* **E1 Lorenz-63**
  (\dot x=\sigma(y-x), \dot y=x(\rho-z)-y, \dot z=xy-\beta z)。
  主眼：短期予測RMSEと長期統計（最大リアプノフ指数）の一致。

* **E2 Mass–Spring–Damper with Input**
  (\ddot x+2\zeta\omega\dot x+\omega^2 x = b u(t))。
  学習入力：PRBS。評価入力：サイン掃引・チャープ・ステップ。
  主眼：FRF（Bodeゲイン・位相）の一致、未見入力での一般化。

* **E3 Burgers 1D**
  (u_t+uu_x=\nu u_{xx})、周期境界。空間差分離散化。
  主眼：波数スペクトル (E(k)) と時間平均エネルギーの一致。空間解像度外挿も可能。

* **E4 Dry Friction Oscillator**
  (m\ddot x+c\dot x+kx+\mu g \tanh(\kappa \dot x)=0)。
  主眼：(\kappa) 増加による劣化の可視化、適用域の明確化。

---

## 3. 新たなモデル定義

* **sindyc_stlsq**：入力あり SINDy。辞書は状態＋入力。
* **edmd**：EDMD/EDMDc。離散モデル（`predict_next` 実装、`predict_derivative` は NotImplementedError）。
* **sindy_pi**：SINDy-PI。積分形式で回帰。
* **sindy_implicit**：implicit-SINDy。SVDによる暗黙式。
* **zero**：基準ゼロモデル。

---

## 4. 実験設定（YAML例）

### E1: Lorenz-63

```yaml
system: lorenz63
params: {sigma: 10.0, rho: 28.0, beta: 2.6667}
r_list: [0.1, 0.25, 0.5]
noise: {SNR_dB: [30, 10]}
splits: {train: 0.6, val: 0.2, test: 0.2}
```

### E2: Mass–Spring–Damper

```yaml
system: mass_spring_input
params: {zeta: 0.02, omega: 6.283, b: 1.0, input_mode: prbs}
r_list: [0.1, 0.25, 0.5, 0.9]
noise: {SNR_dB: [30, 20]}
```

### E3: Burgers 1D

```yaml
system: burgers1d
params: {nu: 0.00318, Nx: 128}
r_list: [0.1, 0.05]
noise: {SNR_dB: [30, 20]}
```

### E4: Dry Friction

```yaml
system: dry_friction_oscillator
params: {m: 1.0, k: 39.48, zeta: 0.02, mu: 0.2, g: 9.81, kappa: 20}
r_list: [0.1, 0.25, 0.5]
noise: {SNR_dB: [30, 10]}
```

---

## 5. 実行手順

1. 必要ライブラリをインストール（`numpy`, `matplotlib`, `pyyaml`）。
2. 実験を実行：

   ```bash
   python -m dynid_benchmark.runners.run_experiment \
     --config exp/E1_Lorenz63.yaml \
     --outdir runs_core4 \
     --models sindy_stlsq,edmd,sindy_pi,sindy_implicit,zero
   ```
3. 全実験をまとめて実行（Core-4 一括スクリプト）：

   ```bash
   python -m paper_core4.make_core4 --outdir runs_core4
   ```
4. 出力確認：`runs_core4/` 以下に図1–4と表1が生成される。

---

## 6. 作業チェックリスト

* [ ] Lorenz63 / Mass–Spring–Damper / Burgers / DryFriction の **systems 実装**。
* [ ] `sindyc_stlsq`, `edmd`, `sindy_pi`, `sindy_implicit`, `zero` の **models 実装**。
* [ ] YAML（E1〜E4）を `exp/` に配置。
* [ ] `run_experiment.py` で共通インタフェースに従い学習・予測・保存。
* [ ] `make_core4.py` で図表（Fig1a–4, Table1）を一括生成。
* [ ] 可視化は**線種×マーカー**を自動切換、色に依存しない。
* [ ] EDMD は等間隔チェックを行い、条件を満たさなければ `error_edmd.txt` を出力して比較から除外。
