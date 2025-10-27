# Agent Operating Notes

このドキュメントは、共同研究エージェントが PyKoopman 系ワークフローを中心に一貫した運用ができるようにするためのメモです。

> ドキュメント類は可能な限り日本語で整備してください（共同研究者間での共有を円滑にするため）。

## リポジトリ概要
- コアライブラリは `dynid_benchmark/` にまとまっており、Poetry（`pyproject.toml`）でパッケージ管理されています。
- 実験設定は `exp/` 以下の YAML で定義し、`dynid_benchmark.runners.run_experiment` から実行します。
- 生成物は `runs*/` や `results/` に保存されるため、大きなファイルはコミット対象に含めないでください。

## PyKoopman ベースの標準フロー
1. Python 3.10+ の仮想環境を用意し、`poetry install` で依存を導入します（SciPy ≤1.11.2 + PyKoopman 1.1 + torch 2.1 系が既定）。
2. 作業前後に `poetry run pytest -q` でスモークテストを実行し、回帰を確認します。
3. 代表的な実験は次のコマンドで起動します。

   ```bash
   poetry run python -m dynid_benchmark.runners.run_experiment \
     --config exp/A1_kappa_sweep.yaml \
     --models pykoopman_edmd,zero \
     --outdir runs
   ```

4. PyKoopman が等間隔サンプリングを要求する点に注意し、`exp/` 側のサンプリング設定と制御入力の長さを常に整合させてください。
5. 学習失敗時は `runs/<exp_id>/<tag>/error_pykoopman_*.txt` を確認し、条件数やサンプリング設定を見直します。

## ノイズ設計ポリシー
- 初動検証は **ノイズなし（`SNR_dB = null`）** を既定とし、モデルの健全性を確認してからノイズケースへ展開します。
- YAML の `noise.SNR_dB` は `[null, ...]` で開始し、高 SNR → 低 SNR の順で段階的に検証します。
- ノイズ条件を追加した場合は、実験メモに意図と結果を残してください。

## コーディングとドキュメント整備
- 副作用はランナー層に閉じ込め、純粋関数と明示的な設定を優先します。
- Python ファイルを編集した際は `poetry run black .` で整形し、ロジックが複雑な箇所のみ簡潔なコメントを付けます。
- 図表は `dynid_benchmark/io/viz.py` の方針に従い、線種 × マーカーで色非依存の可視化を維持します。

## 環境プロファイル
- ルート Poetry プロジェクト: PyKoopman スタックを標準採用（SciPy ≤1.11.2, torch 2.1 系）。
- `envs/pykoopman`: ルート環境と同構成を固定化した再現用プロファイル（コンテナ / CI 向け）。
- SINDy 系ユースケースは別扱いとし、詳細は `docs/sindy_notes.md` を参照してください。

## 外部ライブラリの追加
- 依存を増やす際は `docs/NEW_LIBRARIES.md` を参照し、Poetry ロックファイルへの影響を確認したうえで導入します。
- `NEW_LIBRARIES.md` に追記する場合は、PyKoopman スタックへの影響とテスト手順を明文化してください。

## 成果物とクリーンアップ
- 生成物は `runs/<exp_id>/<tag>/` に集約されます。`metrics_*.json` と `rollout_*.png` で結果を確認し、不要な大容量ファイルは削除またはアーカイブします。
- タスク完了時は、整形とテストの実行ログを残し、必要であれば `docs/` 配下の運用メモを更新してください。
