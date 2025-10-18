# Agent Operating Notes

This document captures working practices and project-specific context so that coding agents can operate consistently across sessions.

> ドキュメント類は可能な限り日本語で整備してください（共同研究者間での共有を円滑にするため）。

## Repository Overview
- Core library lives under `dynid_benchmark/` and is packaged via Poetry (`pyproject.toml`).
- Experiments and configuration files are kept in `exp/` and runnable through `dynid_benchmark.runners.run_experiment`.
- Results artifacts collect under `runs*/` and `results/`; avoid committing large generated files.

## Daily Workflow
1. Create or reuse a Python 3.10+ virtual environment.
2. Install dependencies with `poetry install`; prefer `poetry run <command>` for invoking tools.
3. Run tests with `poetry run pytest` before and after significant changes.
4. Keep notebooks and ad-hoc scripts inside `exp/` or dedicated scratch directories.

## Coding Conventions
- Favor pure functions and explicit configs; keep stateful side effects localized in runners.
- Add succinct comments when logic is not self-evident, but keep the codebase clean.
- Follow Black-compatible formatting and run `poetry run black .` when touching Python files.

## Version Control
- Branch per feature/bugfix; write descriptive commit messages.
- Do not revert user-authored changes; coordinate if the working tree is dirty.
- Large data files belong in external storage; use `.gitignore` for transient outputs.

## Numerical Experiments
- Execute experiments with `poetry run python -m dynid_benchmark.runners.run_experiment --config <yaml> --models <comma-separated models> --outdir runs`.
- Configuration files reside in `exp/`; adjust runtime parameters there before launching.
- Results (metrics, rollouts, data) appear under `runs/<exp_id>/<tag>/`; archive or clean up as needed.

### ノイズ設計ポリシー
- 初動の検証は **ノイズなし（SNR_dB = null）** を既定とし、モデルの健全性を確認してからノイズ付与ケースへ展開する。
- YAML の `noise.SNR_dB` は `[null, ...]` で始め、必要に応じて高 SNR→低 SNR へ段階的に評価する。
- ノイズを追加する場合も、意図を実験メモに残し、結果が解釈できるよう逐次記録する。

## External Libraries / 追加ライブラリ
- `pykoopman` を既定依存として導入済み。`pykoopman_edmd` / `pykoopman_edmdc` モデルを通じて Koopman 系（EDMD/EDMDc）ベースラインを利用でき、失敗時は `error_pykoopman_*.txt` に詳細が保存される。
- `pysindy` は Poetry グループ `pysindy` として任意導入。SINDy-PI（`cvxpy` 依存）と PyKoopman は SciPy 要件が異なるため、目的に応じて環境を切り替える。
- 追加ライブラリは作業前に `NEW_LIBRARIES.md` のロードマップを確認し、Poetry ロックファイルへの影響を把握してから導入すること。

## Environment Profiles
- ルート Poetry プロジェクト: SciPy ≤1.11.2 + PyKoopman 1.1 + torch 2.1 系の標準スタック。Koopman 系実験はこちらを既定とする。
- `envs/pysindy`: 最新スタック（NumPy/SciPy 2024 系）＋ PySINDy 用。SINDy-PI（`cvxpy` 使用）など PyKoopman と両立しない検証向け。
- `envs/pykoopman`: ルート環境と同等構成を再現するための固定プロファイル。コンテナや CI での再現性確保に利用。
