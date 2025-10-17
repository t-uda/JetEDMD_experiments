# PyKoopman 環境メモ

このサブプロジェクトは PyKoopman v1.1.0 を利用するための Poetry 定義です。

## 目的
- SciPy ≤1.11.2 / NumPy ≤1.26 / scikit-learn 1.1.3 といった旧世代スタックで JetEDMD を実行する。
- `torch 2.1.2` + `torchvision 0.16.0` + `torchaudio 2.1.2`（CPU 版）を利用する前提。
- Python 3.10 / 3.11 を Singularity 上で用意し、その中で `poetry install` を行う。

## 使い方
1. Singularity 内で Python 3.10 系を有効化する。
2. `cd envs/pykoopman`
3. 必要に応じて `poetry env use /path/to/python3.10`
4. `POETRY_VIRTUALENVS_IN_PROJECT=true poetry lock`（初回のみ）
5. `POETRY_VIRTUALENVS_IN_PROJECT=true poetry install`

> **注意**: このリポジトリ上では Python 3.13 を利用しているため、torch 2.1 系の wheel が提供されず `poetry lock` が失敗します。必ず Singularity など PyTorch がサポートする Python バージョンからロックファイルを生成してください。

## テスト
- `poetry run pytest -q`（JetEDMD のテストが SciPy 1.11 系で動作することを要確認）
- CI や `tox` からは `tox -e pykoopman` のように呼び出すことを想定。
