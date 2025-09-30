# 外部ライブラリ採用チェック（SINDy 系）

## 採用候補ライブラリ一覧

* **PySINDy**：SINDy（STLSQ/SR3等）、SINDYc（入力付き）、**SINDy-PI**、PDE-FIND ほか。チュートリアルとAPIが充実。([pysindy.readthedocs.io][1])
* **PyKoopman**：Koopman 演算子近似の総合実装。**EDMD/EDMDc** を scikit-learn 風に提供。([pykoopman.readthedocs.io][2])
* **PyDMD**：DMD 系（Extended DMD, DMDc 含む）。EDMD 比較の軽量ベースラインとして有用。([pydmd.github.io][3])

---

## このプロジェクト内での使い方と注意点

複数の実装は必要に応じて名前文字列で dispatch 切り替えできるようにすること．比較デバッグの目的で，現在の教育実装も原則として残すこと．ただし，デフォルトは高信頼の外部ライブラリ版を優先して採用する．

### PySINDy

* **用途**：本テンプレの教育実装（SINDy-STLSQ／SINDy-PI）を**高信頼版**に置換。入力付き同定（SINDYc）や PDE-FIND も拡張候補。
* **注意**：

  * **差分微分**が前提の設定では粗サンプリング時に脆い → **SINDy-PI**（積分形式）を優先候補に。([pysindy.readthedocs.io][4])
  * 最適化器や微分器の選択（SR3/Smoothed FD など）で性能が変わるため、**実験 YAML 側でハイパ表示**を推奨。([pysindy.readthedocs.io][5])
  * 追加最適化（SR3/L0/ベイズ）や凸最適化系は**追加エクストラ**が必要（例：`cvxpy` 等）。CI では軽量設定を既定に。([pysindy.readthedocs.io][1])

### PyKoopman

* **用途**：既存の **EDMD/EDMDc** 教育実装を置換し、**離散時間 one-step 予測**でテンプレの `predict_next` 経路に統合。
* **注意**：

  * **等間隔サンプリング**前提（本テンプレの EDMD と同じ制約）；ジッタ・欠測がある設定では**自動でスキップ**または**事前リサンプリング**。([pykoopman.readthedocs.io][2])
  * ドキュメント推奨の開発セット（GPU対応含む）は**任意**。実験は標準パッケージで十分。([pykoopman.readthedocs.io][2])

### PyDMD

* **用途**：**DMD/Extended DMD/DMDc** による Koopman 系の**軽量ベースライン**。時系列埋め込みやモード可視化の補助に。
* **注意**：

  * 主眼が EDMD なら PyKoopman を優先、PyDMD は**比較用**として最小限採用。([pydmd.github.io][3])

---

## 出典（公式ドキュメント等）

* PySINDy：総合ドキュメント／インストール／機能一覧、**SINDy-PI チュートリアル**。([pysindy.readthedocs.io][1])
* PyKoopman：ドキュメント／パッケージ情報／導入メモ（dev セットは任意）。([pykoopman.readthedocs.io][2])
* PyDMD：公式ドキュメント／PyPI の説明。([pydmd.github.io][3])

---

# エージェント向け・短い指示（導入フローのみ）

1. **PySINDy** を追加し、テンプレの SINDy／SINDy-PI を**切替比較**できるようアダプタモデルを 1 本追加（最小構成：多項式＋任意 Fourier、微分器/最適化器は引数）。SINDy-PI を**粗サンプリング時の既定**に。([pysindy.readthedocs.io][1])
2. **PyKoopman** を追加し、EDMD/EDMDc を**離散モデル**として差し替え（`predict_next` 経路を使用）。**等間隔チェック**は既存ロジックを流用。([pykoopman.readthedocs.io][2])
3. **PyDMD** は DMD/EDMD 系の**補助ベースライン**として任意採用（結果表に “DMD/EDMD(alt)” を追加）。([pydmd.github.io][3])

> すべて **図の方針（線種×マーカーで色非依存）** と **失敗時の error ログ保存** を維持。

[1]: https://pysindy.readthedocs.io/?utm_source=chatgpt.com "PySINDy — pysindy 2.0.1.dev15+g472e5a236 documentation"
[2]: https://pykoopman.readthedocs.io/?utm_source=chatgpt.com "PyKoopman — pykoopman 1.1.1 documentation"
[3]: https://pydmd.github.io/PyDMD/?utm_source=chatgpt.com "Welcome to PyDMD's documentation! - GitHub Pages"
[4]: https://pysindy.readthedocs.io/en/stable/examples/9_sindypi_with_sympy/example.html?utm_source=chatgpt.com "SINDy-PI Feature Overview — pysindy 2.0.0 documentation"
[5]: https://pysindy.readthedocs.io/en/latest/examples/index.html?utm_source=chatgpt.com "Tutorials — pysindy 2.0.1.dev15+g472e5a236 documentation"

