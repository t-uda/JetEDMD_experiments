# 外部ライブラリ採用ガイド（PyKoopman 中心）

このファイルは、Poetry 依存に新しいライブラリを追加する際の判断材料をまとめます。SINDy 系の詳細メモは `docs/sindy_notes.md` に移動しました。

## 導入状況（2025-10-17 時点）

| ステータス | ライブラリ | 用途 | 備考 |
| --- | --- | --- | --- |
| 標準 | **PyKoopman 1.1** | EDMD/EDMDc の主力実装。`pyproject.toml` に固定済み。 | 等間隔サンプリング必須。`envs/pykoopman` で再現環境を維持。 |
| 標準 | **PyDMD 0.4.1** | 軽量な Koopman 系比較ベースライン。 | `predict_next` ベースで離散モデルを提供。 |
| 標準 | **torch 2.1 / lightning 2.0.9** | ニューラル Koopman 拡張や器用な最適化に備えた基盤。 | GPU 版が必要な場合は別プロファイルを用意。 |
| 任意 | その他 Koopman/SDE 関連（例：`optht`） | 実験に合わせて導入。 | 追加時は `poetry lock` 差分と CI 時間を確認。 |

> 依存の追加・更新では `poetry update <pkg>` 後に `poetry run pytest -q` を通し、`runs/` 以下の成果物で動作確認を行ってください。

## 導入プロセス

1. **要件整理**：新ライブラリが PyKoopman スタックと整合するか（SciPy バージョン、torch との相性など）を確認。
2. **Poetry 設定**：`pyproject.toml` に依存を追加し、`poetry lock` の差分をレビュー。必要なら extras グループを定義します。
3. **テスト**：`poetry run pytest -q` で既存テストを通し、`dynid_benchmark.runners.run_experiment` で代表 YAML を実行。
4. **ドキュメント更新**：導入意図・検証結果を `docs/` 以下に追記し、既存の運用メモとの整合を図る。

## 評価ポイント

- **サンプリング依存性**：等間隔が必須か、ジッタや欠測を許容するかを明確にする。
- **計算コスト**：学習時間や GPU 依存などを試験し、CI への影響を評価。
- **ロギング**：失敗時に `error_*.txt` を出力・解析できるよう、ラッパーレイヤを整備する。

## 参考

- PyKoopman ドキュメント: <https://pykoopman.readthedocs.io/>
- PyDMD ドキュメント: <https://pydmd.github.io/PyDMD/>
- SINDy 系ライブラリ導入メモ: `docs/sindy_notes.md`
