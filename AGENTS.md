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
