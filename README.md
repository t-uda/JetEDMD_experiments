# dynid_benchmark_template

YAML を読み込み、**データ生成 → 観測化（粗サンプリング・ノイズ等）→ 学習 → 評価 → 図出力**までを一気に行う実験雛形です。

- 8本の代表系（A1/A2/B1/B2/C1-1/C1-2/D1/D2）に対応。
- 追加の手法は `dynid_benchmark/models/base.py` の `Model` を継承し、`MODEL_REGISTRY` に登録するだけ。
- 例として **SINDy（STLSQ）** と **Zero/MeanDerivative** の軽量ベースラインを同梱（外部依存なし）。

## インストール

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
```

> `requirements.txt` は極小（numpy, pyyaml, matplotlib）。SciPy 等は不要です。

## 使い方（A1 例）

```bash
python -m dynid_benchmark.runners.run_experiment --config exp/A1_kappa_sweep.yaml --models sindy_stlsq,zero --outdir runs
```

- 生成物は `runs/<exp_id>/<tag>/` に保存（`metrics_*.json`, `rollout_*.png`, `data_*.npz`）。
- `--time` で総シミュレーション時間を上書き可能。

## 手法の追加

`dynid_benchmark/models/base.py` の `Model` を継承：

```python
from dynid_benchmark.models.base import Model, register_model

@register_model
class YourMethod(Model):
    name = "your_method"
    def fit(self, t, y, u=None):
        # 学習処理
        ...
    def predict_derivative(self, t, x, u=None):
        # 連続時間ベクトル場 f_hat(x[,u]) を返す
        return ...
```

`--models your_method` で呼び出せます。

## 実装上の注意
- 図は **matplotlib** のみ使用（seaborn 不要）。
- 1 図 1 チャートに統一（論文図に貼りやすい）。

## 構成
- `dynid_benchmark/systems/`: 各ベンチマークの真値生成器（ODE/SDE/PDE）
- `dynid_benchmark/models/`: 手法の実装とレジストリ
- `dynid_benchmark/evaluation/`: 指標計算・保存
- `dynid_benchmark/io/`: データ I/O, 可視化
- `dynid_benchmark/runners/`: 実験ランナー
- `exp/`: 実験ごとの YAML
