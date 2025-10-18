# 外部ライブラリ採用チェック（SINDy / Koopman 系）

## 導入状況（2025-10-17 時点）

- **導入済み（標準スタック）**：**PyKoopman 1.1 系** + **PyDMD 0.4.1** + **torch 2.1 系**（`pyproject.toml` にて標準依存。Koopman 系ベンチマークをルート環境で実行可能）。 ([pykoopman.readthedocs.io][2])
- **任意導入（Poetry グループ `pysindy`）**：**PySINDy 1.7 系**（`dynid_benchmark/models/pysindy_adapter.py` としてアダプタ実装）。SINDy-PI は `cvxpy` を追加した環境でのみ有効。 ([pysindy.readthedocs.io][1])
- **選択導入**：その他の Koopman/SINDy 関連ライブラリ（必要性を `NEW_LIBRARIES.md` で検討）。 ([pydmd.github.io][3])

> バージョンや依存追加の判断は Poetry 環境で一元管理する。実験系では `poetry lock` の更新差分を確認しつつ、CI の所要時間を考慮して余計な依存を避ける。

## 採用候補ライブラリ一覧（用途メモ）

* **PySINDy**：SINDy（STLSQ/SR3 等）、SINDYc（入力付き）、**SINDy-PI**、PDE-FIND ほか。チュートリアルと API が充実。([pysindy.readthedocs.io][1])
* **PyKoopman**：Koopman 演算子近似の総合実装。**EDMD/EDMDc** を scikit-learn 風に提供。([pykoopman.readthedocs.io][2])
* **PyDMD**：DMD 系（Extended DMD, DMDc 含む）。EDMD 比較の軽量ベースラインとして有用。([pydmd.github.io][3])

---

## このプロジェクト内での使い方と注意点

複数実装は名前文字列で dispatch 切り替えできるよう統一する。既存の教育実装は比較デバッグ用途で残しつつ、既定値は高信頼の外部ライブラリ版を優先。導入済みライブラリのメンテでは **Poetry での依存固定** と **回帰テスト（`poetry run pytest`）** を必ず通す。

> 実験 YAML の `noise.SNR_dB` は全て `[null, ...]` で始まるよう整備。外部ライブラリ導入時も、まずノイズ無しケースで挙動確認してからノイズ付き検証へ移行すること。

### PySINDy（任意導入）

* **用途**：本テンプレの教育実装（SINDy-STLSQ／SINDy-PI）を高信頼版へ差し替えるオプション。入力付き同定（SINDYc）や PDE-FIND は今後の拡張候補。
* **注意**：

  * 利用時は `poetry install --with pysindy` で依存を追加するか、`envs/pysindy` プロファイルを使用する。
  * **差分微分**が前提の設定では粗サンプリング時に脆い → SINDy-PI（積分形式）を粗サンプル既定候補として `pysindy_pi` で提供。([pysindy.readthedocs.io][4])
  * 最適化器や微分器の選択（SR3/Smoothed FD など）で性能が変わるため、**実験 YAML 側でハイパ表示**を推奨。([pysindy.readthedocs.io][5])
  * 追加最適化（SR3/L0/ベイズ）や凸最適化系は **追加依存（例：`cvxpy`）** が必要。`cvxpy` は SciPy ≥1.13 を要求するため、PyKoopman（SciPy ≤1.11.2 依存）と同居させる場合は環境を分けること。
  * `pysindy` 本体で `numpy.math` 利用が残っているため、NumPy 2.0 互換パッチ（`np.math = math`）を入れている。将来バージョンアップ時には再確認が必要。

### PyKoopman

* **用途**：`pykoopman_edmd` / `pykoopman_edmdc` を通じて PyKoopman 実装の EDMD/EDMDc を利用し、教育実装との精度比較や制御入力付きケースを検証する。
* **注意**：

  * ルート環境は **SciPy ≤1.11.2 + torch 2.1** でロックしている。SINDy-PI（`cvxpy` 依存）など SciPy ≥1.13 を要求するケースとは環境を分離する。
  * **等間隔サンプリング**前提（本テンプレの EDMD と同じ制約）；ジッタ・欠測がある設定では**自動でスキップ**または**事前リサンプリング**。([pykoopman.readthedocs.io][2])
  * ドキュメント推奨の開発セット（GPU 対応含む）は任意。Poetry では CPU 版のみ導入する想定。([pykoopman.readthedocs.io][2])
  * `torch` 系依存は CPU 版ホイールを標準リポジトリから取得している。GPU 版が必要な場合は別プロファイルを検討する。

#### PyKoopman 実装メモ

1. **依存管理**
   - ルート `pyproject.toml` で `pykoopman 1.1.0` と付随する Koopman スタックを固定済み。`envs/pykoopman` では同構成をコンテナ/CI 用に再現する。
   - `cvxpy` を導入する場合（SINDy-PI 利用時）は SciPy を ≥1.13 に戻す必要があり、PyKoopman と同居できないため `pysindy` グループや別環境で切り替える。
2. **アダプタ構成**
   - `dynid_benchmark/models/pykoopman_adapter.py` で `PyKoopmanEDMD` / `PyKoopmanEDMDc` を提供。多項式観測（`Polynomial`）＋ `Koopman(regressor=EDMD/EDMDc)` を利用し、等間隔チェックや制御入力の整合性検証を実装済み。
   - 学習失敗時はランナーが `error_pykoopman_*.txt` を保存し、ログから原因を追跡できる。
3. **ランナー統合**
   - `run_experiment.py` の既定モデルを `pykoopman_edmd,zero` に変更済み。制御付きケースでは `--models pykoopman_edmdc,zero` を利用し、`exp/` YAML 側で入力系列を生成する。
   - YAML カスタムを追加する際はサンプリング設定と制御入力の長さが整合しているか確認する。
4. **テスト**
   - `tests/test_models_pykoopman.py` で EDMD/EDMDc の回帰精度と不等間隔エラーをカバー。PyKoopman 未導入環境では `importorskip` により自動スキップ。
   - 追加の長期ロールアウトやノイズ混入ケースは今後の拡張候補。

### PyDMD

* **用途**：**DMD/Extended DMD/DMDc** による Koopman 系の**軽量ベースライン**。時系列埋め込みやモード可視化の補助に。
* **注意**：

  * 主眼が EDMD なら PyKoopman を優先、PyDMD は比較用として最小限採用。([pydmd.github.io][3])
  * 可視化 API（モード図）を使う場合は Matplotlib 依存が増える。CI 負荷を考慮し、当面は数値評価に限定する。

---

## 出典（公式ドキュメント等）

* PySINDy：総合ドキュメント／インストール／機能一覧、**SINDy-PI チュートリアル**。([pysindy.readthedocs.io][1])
* PyKoopman：ドキュメント／パッケージ情報／導入メモ（dev セットは任意）。([pykoopman.readthedocs.io][2])
* PyDMD：公式ドキュメント／PyPI の説明。([pydmd.github.io][3])

---

# エージェント向け・短い指示（ライブラリ導入ロードマップ）

1. **PySINDy（完了）**：`pysindy` / `pysindy_pi` モデルが実験ランナーから呼び出せることを確認済み。フォローアップとして
   - `poetry run pytest` で回帰を取り、NumPy 2.0 互換パッチが効いているかを継続確認すること。
   - `cvxpy` なし環境では `pysindy_pi` を自動スキップするため、テストの `xfail` 条件とドキュメントを同期させること。
   - 新規 YAML を追加する場合は `optimizer`, `differentiation_method` を明示し、再現可能性を担保する。
2. **PyKoopman（フォローアップ）**：アダプタ・テストは実装済み。追加で以下を検討する。
   - 多項式観測の正規化／条件数モニタリングを導入し、数値不安定時のログを強化する。
   - C1 系の入力有り YAML に `pykoopman_edmdc` を組み込み、FRF/Bode など入力一般化評価を追加する。
   - 長期ロールアウトやノイズ混入ケースのベンチマークを追加し、結果をドキュメント化する。
3. **PyDMD（任意導入）**：軽量比較用として `dynid_benchmark/models/pydmd_adapter.py`（仮）を追加。
   - **タスク**：Poetry へ依存追加（必要であれば extras で切り分け）。`predict_next` ベースで DMD/EDMD をサポートし、図表には "DMD/EDMD(alt)" を追加。
   - **留意点**：モード可視化など重たい機能はオプション扱い。既存 CI に影響を与えない構成で導入する。

> すべて **図の方針（線種×マーカーで色非依存）** と **失敗時の error ログ保存** を維持。

[1]: https://pysindy.readthedocs.io/?utm_source=chatgpt.com "PySINDy — pysindy 2.0.1.dev15+g472e5a236 documentation"
[2]: https://pykoopman.readthedocs.io/?utm_source=chatgpt.com "PyKoopman — pykoopman 1.1.1 documentation"
[3]: https://pydmd.github.io/PyDMD/?utm_source=chatgpt.com "Welcome to PyDMD's documentation! - GitHub Pages"
[4]: https://pysindy.readthedocs.io/en/stable/examples/9_sindypi_with_sympy/example.html?utm_source=chatgpt.com "SINDy-PI Feature Overview — pysindy 2.0.0 documentation"
[5]: https://pysindy.readthedocs.io/en/latest/examples/index.html?utm_source=chatgpt.com "Tutorials — pysindy 2.0.1.dev15+g472e5a236 documentation"
