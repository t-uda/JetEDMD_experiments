# envs ディレクトリについて

このディレクトリは数値実験用の **依存セット別 Poetry プロジェクト** をまとめる場所です。

- `pysindy/`: 最新 NumPy/SciPy 系（2024 時点）＋ PySINDy を用いるオプション環境。SINDy-PI（`cvxpy`）など PyKoopman と両立しない検証用。
- `pykoopman/`: ルート環境と同一の PyKoopman スタック（SciPy ≤1.11.2, torch 2.1 系など）を固定化した再現プロファイル。PySINDy 系とは分離して利用します。

各サブディレクトリは独立した `pyproject.toml` / `poetry.lock` を持ち、以下の用途で利用します。

1. Singularity コンテナ内で `poetry install` して環境を再現。
2. `tox` / CI で `tox -e pysindy` / `tox -e pykoopman` のようにテストを切り替え。
3. ベースリポジトリのソースコード (`dynid_benchmark`) はパス依存 `{ path = "../..", develop = true }` で読み込みます。

> **注意**: ここでの Poetry プロジェクトは実験用スタック管理に限定し、ルートの `pyproject.toml`（メタデータ用）とは役割を分担しています。
