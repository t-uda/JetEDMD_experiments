# 外部ライブラリ採用チェック（SINDy / Koopman 系）

## 導入状況（2025-10-17 時点）

- **導入済み**：**PySINDy 1.7 系**（`pyproject.toml` に追記済み、`dynid_benchmark/models/pysindy_adapter.py` としてアダプタ実装）。標準 SINDy は `pysindy` モデルとして登録済み。SINDy-PI は `cvxpy` を追加導入した環境でのみ利用可能（`pyproject.toml` からは除外済み）。 ([pysindy.readthedocs.io][1])
- **次タスク**：**PyKoopman**（EDMD/EDMDc/Koopman Operator）。`predict_next` ベースの離散モデルとしてアダプション予定。 ([pykoopman.readthedocs.io][2])
- **選択導入**：**PyDMD**（DMD/EDMD 系ベースライン）。Koopman 系比較用途で必要なら追加。 ([pydmd.github.io][3])

> バージョンや依存追加の判断は Poetry 環境で一元管理する。実験系では `poetry lock` の更新差分を確認しつつ、CI の所要時間を考慮して余計な依存を避ける。

## 採用候補ライブラリ一覧（用途メモ）

* **PySINDy**：SINDy（STLSQ/SR3 等）、SINDYc（入力付き）、**SINDy-PI**、PDE-FIND ほか。チュートリアルと API が充実。([pysindy.readthedocs.io][1])
* **PyKoopman**：Koopman 演算子近似の総合実装。**EDMD/EDMDc** を scikit-learn 風に提供。([pykoopman.readthedocs.io][2])
* **PyDMD**：DMD 系（Extended DMD, DMDc 含む）。EDMD 比較の軽量ベースラインとして有用。([pydmd.github.io][3])

---

## このプロジェクト内での使い方と注意点

複数実装は名前文字列で dispatch 切り替えできるよう統一する。既存の教育実装は比較デバッグ用途で残しつつ、既定値は高信頼の外部ライブラリ版を優先。導入済みライブラリのメンテでは **Poetry での依存固定** と **回帰テスト（`poetry run pytest`）** を必ず通す。

### PySINDy（導入済み）

* **用途**：本テンプレの教育実装（SINDy-STLSQ／SINDy-PI）を高信頼版へ差し替え済み。入力付き同定（SINDYc）や PDE-FIND は今後の拡張候補。
* **注意**：

  * **差分微分**が前提の設定では粗サンプリング時に脆い → SINDy-PI（積分形式）を粗サンプル既定候補として `pysindy_pi` で提供。([pysindy.readthedocs.io][4])
  * 最適化器や微分器の選択（SR3/Smoothed FD など）で性能が変わるため、**実験 YAML 側でハイパ表示**を推奨。([pysindy.readthedocs.io][5])
  * 追加最適化（SR3/L0/ベイズ）や凸最適化系は **追加依存（例：`cvxpy`）** が必要。`cvxpy` は SciPy ≥1.13 を要求するため、PyKoopman（SciPy ≤1.11.2 依存）と同居させる場合は環境を分けること。
  * `pysindy` 本体で `numpy.math` 利用が残っているため、NumPy 2.0 互換パッチ（`np.math = math`）を入れている。将来バージョンアップ時には再確認が必要。

### PyKoopman

* **用途**：既存の **EDMD/EDMDc** 教育実装を置換し、離散時間 one-step 予測でテンプレの `predict_next` 経路に統合。
* **注意**：

  * **等間隔サンプリング**前提（本テンプレの EDMD と同じ制約）；ジッタ・欠測がある設定では**自動でスキップ**または**事前リサンプリング**。([pykoopman.readthedocs.io][2])
  * ドキュメント推奨の開発セット（GPU 対応含む）は任意。Poetry では CPU 版のみ導入する想定。([pykoopman.readthedocs.io][2])
  * 追加依存（`torch`, `cvxpy` など）を要求する機能が多いため、Poetry への一括導入は慎重に行う。PyKoopman 1.1.0 自体が **SciPy ≤1.11.2** を必須とし、`torch` もビルド環境依存が大きい。

#### PyKoopman 導入メモ（最小構成）

1. **依存管理**
   - `pykoopman 1.1.0` は **SciPy (>1.6, ≤1.11.2)** と CPU 版 `torch` を依存として要求する。既存環境（NumPy 2.x / SciPy 1.15 / cvxpy 1.7.3）とは競合するため、Poetry への直接追加は失敗する。
   - 対応案：
     - `pykoopman` 専用のサブ環境（例：`poetry env use` で仮想環境を複製し、`poetry add --lock pykoopman^1.1` を実行）を作る。
     - もしくは Pipenv/venv など別管理で PyKoopman 実験系を分離し、ベース環境との混在を避ける。
   - 本リポジトリでは `envs/pykoopman` に専用の `pyproject.toml` を配置し、Singularity 等で個別にロックを生成する方針。
   - `cvxpy` を導入する場合（SINDy-PI 利用時）は SciPy を ≥1.13 に戻す必要があり、PyKoopman と同居できない。
2. **アダプタ実装**
   - `dynid_benchmark/models/pykoopman_adapter.py`（仮）を新設し、`Model` を継承した `PyKoopmanEDMDModel` / `PyKoopmanEDMDcModel` を定義。
   - `fit` で `pykoopman.regression.EDMD` / `EDMDc` を初期化。辞書関数は既存の多項式＋Fourier を流用（`PolynomialLibrary` 相当を PyKoopman の `Observables` API で再現）。
   - 差分時間 `dt` が一定でない場合は早期例外。`predict_next` を実装し、ランナーの離散モデル経路を使用。
   - コントロール付き/なしを同一クラスで扱うなら `u` の有無で `EDMD` ↔ `EDMDc` を切り替える。
3. **ランナー統合**
   - `dynid_benchmark/models/__init__.py` と `run_experiment.py` のデフォルト候補へモデルキーを追加（例：`pykoopman_edmd`）。
   - 実験 YAML（特に C1 系）に `pykoopman_edmd` を追加する際は、入力有無とサンプリング設定が一致するよう注意。
4. **エラーハンドリング**
   - 等間隔チェック失敗時は `RuntimeError` で `error_pykoopman*.txt` にメッセージを残す。
   - import ガードで未インストール時は明確な案内を表示（`poetry add pykoopman`）。

##### テスト計画

- `tests/test_models.py` に PyKoopman 用のパラメタ化ケースを追加。
  * 入力なし短尺データで `fit→rollout` が例外なく完了するか。
  * 入力あり（`EDMDc` 相当）で `u` を渡し、`predict_next` が動作するか。
  * 不等間隔サンプルを渡して期待通りエラーになるか（`xfail` または `pytest.raises`）。
- `tests/test_smoke.py` で import 可能かを確認し、未導入環境では `pytest.skip`。
- `poetry run pytest -q` を CI やローカルで回し、PyKoopman 追加後の実行時間をモニタ。必要なら `slow` マークで制御。
- 可能であれば `exp/A1_kappa_sweep.yaml` などを用いた簡易ラン（`--models pykoopman_edmd,zero`）を README か手順に追記し、ローカル検証のルーチン化を図る。

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
2. **PyKoopman（次アクション）**：離散モデル `predict_next` 互換のアダプタを `dynid_benchmark/models/pykoopman_adapter.py`（仮）として実装。
   - **タスク**：Poetry へ依存追加 → import ガード付きアダプタ → `tests/test_models.py` へパラメトリックテスト → `--models pykoopman_edmd` 追加。
   - **留意点**：等間隔サンプリング検証を再利用し、欠測ケースでは `error_pykoopman.txt` を確実に吐く。
3. **PyDMD（任意導入）**：軽量比較用として `dynid_benchmark/models/pydmd_adapter.py`（仮）を追加。
   - **タスク**：Poetry へ依存追加（必要であれば extras で切り分け）。`predict_next` ベースで DMD/EDMD をサポートし、図表には "DMD/EDMD(alt)" を追加。
   - **留意点**：モード可視化など重たい機能はオプション扱い。既存 CI に影響を与えない構成で導入する。

> すべて **図の方針（線種×マーカーで色非依存）** と **失敗時の error ログ保存** を維持。

[1]: https://pysindy.readthedocs.io/?utm_source=chatgpt.com "PySINDy — pysindy 2.0.1.dev15+g472e5a236 documentation"
[2]: https://pykoopman.readthedocs.io/?utm_source=chatgpt.com "PyKoopman — pykoopman 1.1.1 documentation"
[3]: https://pydmd.github.io/PyDMD/?utm_source=chatgpt.com "Welcome to PyDMD's documentation! - GitHub Pages"
[4]: https://pysindy.readthedocs.io/en/stable/examples/9_sindypi_with_sympy/example.html?utm_source=chatgpt.com "SINDy-PI Feature Overview — pysindy 2.0.0 documentation"
[5]: https://pysindy.readthedocs.io/en/latest/examples/index.html?utm_source=chatgpt.com "Tutorials — pysindy 2.0.1.dev15+g472e5a236 documentation"
