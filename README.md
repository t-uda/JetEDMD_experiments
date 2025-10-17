# JetEDMD 実験リポジトリ

`dynid_benchmark` を基盤に、力学系同定の数値実験を再現可能な形で実施するためのテンプレートです。基準ベンチマークの実装、YAML 設定によるパイプライン制御、学習モデルの追加手順、成果物管理の流れをまとめています。

## 特徴
- A1〜E4 の系列（ODE / SDE / PDE / ハイブリッド）をカバーする代表ベンチマークを同梱。
- `exp/` 配下の YAML で「データ生成 → 観測化 → 学習 → 評価 → 図出力」までを一括実行。
- `dynid_benchmark.models.register_model` による軽量なプラグイン API で手法を追加可能。
- 実行環境は Poetry で管理し、依存関係は `numpy`, `pyyaml`, `matplotlib` に加え **PyKoopman / PyDMD / torch** を標準採用（SINDy 系を使う場合はオプション `pysindy` グループを有効化）。

## クイックスタート
0. 必要要件：Python 3.10 以上、Poetry がインストール済みであること。
1. 依存関係の導入とスモークテスト実行：

```bash
poetry install
poetry run pytest -q
```

2. 実験の起動例（A1 κ スイープ、EDMD + ベースライン Zero モデル）：

```bash
poetry run python -m dynid_benchmark.runners.run_experiment \
  --config exp/A1_kappa_sweep.yaml \
  --models edmd,zero \
  --outdir runs
```

生成された成果物は `runs/<exp_id>/<tag>/` に保存され、メトリクス（`metrics_*.json`）、ロールアウト図（`rollout_*.png`）、シリアライズ済みデータ（`data_*.npz`）などを含みます。`--time` でシミュレーション時間の上書き、`--models` に複数モデルをカンマ区切りで指定できます。

## プロジェクト構成
- `dynid_benchmark/`：真値システム、モデル実装、評価指標、I/O、ランナー等のコアライブラリ。
- `exp/`：実験設定 YAML。一連のパラメータ調整はここで行います。
- `results/`, `runs*/`：実験結果の保存先。大きな成果物はリポジトリにコミットしないよう注意。
- `tests/`：回帰テスト・スモークテスト。
- `envs/`：Poetry で管理する環境プロファイル（PySINDy 用 / PyKoopman 用）。
- `AGENTS.md`, `INSTRUCTIONS.md`：共同研究・エージェント向けの運用メモ。

## 環境プロファイルの切り替え

Singularity などで依存セットを切り替えやすくするため、`envs/` 以下に用途別の Poetry プロジェクトを用意しています。

| プロファイル | Python | 主な依存 | 用途 |
| --- | --- | --- | --- |
| プロジェクトルート | `3.10–3.11` | SciPy ≤1.11.2 + PyKoopman 1.1 + torch 2.1 | 標準（Koopman 系） |
| `envs/pysindy` | `>=3.10,<4.0` | 最新版 SciPy 系 + PySINDy | SINDy 系比較、SINDy-PI（`cvxpy`）検証 |
| `envs/pykoopman` | `3.10–3.11` | ルートと同構成（固定ロック） | コンテナ / CI 用の再現プロファイル |

### 利用例（PyKoopman スタック）

```bash
cd envs/pykoopman
poetry env use /usr/bin/python3.10        # PyTorch 2.1 が対応する Python を指定
POETRY_VIRTUALENVS_IN_PROJECT=true poetry lock
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install
poetry run pytest -q                      # 任意: テスト確認
```

PySINDy 用の `envs/pysindy` でも同様に `poetry install` を実行します。どちらの環境でもリポジトリ本体 (`dynid_benchmark`) はパス依存 `{ path = "../.." }` で共有されるため、ソースコードは一箇所で保守できます。

## ベンチマーク系列の概要
- **A 系列**：乾燥摩擦振動子（A1）やバウンシングボール（A2）など、非滑らかさやイベントを含む低次元力学系。粗いサンプリングでの挙動再現性を検証します。
- **B 系列**：Ornstein–Uhlenbeck（B1）や二井戸 SDE（B2）など確率的システム。ノイズ存在下での統計量一致と長時間挙動の安定性を評価します。
- **C 系列**：外部入力を持つ質量–ばね系（C1）や Duffing 強制振動など、未見入力への一般化性能と周波数応答を確認します。
- **D 系列**：1 次元 Burgers（D1）、Kuramoto–Sivashinsky（D2）といった PDE の半離散 ODE。空間スペクトルやエネルギー指標で評価します。
- **E 系列**：Lorenz-63 カオス（E1）、入力付き LTI 系（E2）、Burgers 拡張（E3）、乾燥摩擦コア（E4）など、Core-4 実験を中心とした比較用セット。長期統計や未見入力一般化を重視します。

## モデルの追加方法

`dynid_benchmark.models.base.Model` を継承し、`@register_model` デコレータでレジストリ登録します。以下は最小例です。

```python
from dynid_benchmark.models.base import Model, register_model

@register_model
class YourMethod(Model):
    name = "your_method"

    def fit(self, t, y, u=None):
        ...  # 学習ロジック

    def predict_derivative(self, t, x, u=None):
        return ...  # 連続時間ベクトル場 f_hat(x[, u])
```

登録後は `--models your_method` で実験ランナーから呼び出せます。実装例は `dynid_benchmark/models/`（SINDy-STLSQ, SINDy-PI, EDMD, Zero/MeanDerivative など）を参照してください。

## 外部ライブラリ版 SINDy の利用

`pysindy` パッケージをラップしたモデルを追加済みです。利用時は `poetry install --with pysindy` で依存を追加するか、`envs/pysindy` プロファイルを使用してください。積分形式の `pysindy_pi` は追加依存 `cvxpy`（SciPy ≥1.13）が必要で、PyKoopman 1.1 系（SciPy ≤1.11.2）とは同一環境で共存できません。粗サンプリングでの高精度化が必要な際には別環境で `cvxpy` を整えた上で `--models pysindy_pi,zero` に切り替え、失敗時は教育実装 `sindy_pi` の併用も検討してください。教育実装（`sindy_stlsq`, `sindy_pi`）との比較は `--models` で手動切り替えできます。

## 開発のヒント
- 可能な限り純粋関数と明示的な設定を用い、副作用はランナーに閉じ込めてください。
- Python ファイルを編集した際は `poetry run black .` で整形し、ドキュメントは原則として日本語で記述します。
- 主要な変更前後には `poetry run pytest` を実行し、挙動退行をチェックします。
- 探索的なノートブックやスクリプトは `exp/` あるいは専用のスクラッチディレクトリに配置してください。
- `runs/` や `results/` に蓄積された大型ファイルは整理・アーカイブし、リポジトリを軽量に保ちます。
