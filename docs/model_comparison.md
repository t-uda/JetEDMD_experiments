# モデル比較ランナー構成メモ

## 目的と概要
- `dynid_benchmark.runners.compare_models` は，観測点数（`r_list`）を変化させながら複数モデルのロールアウト性能を一括で評価し，可視化するためのランナーです。
- 既存の `run_experiment` の処理を踏襲しつつ，モデル別に RMSE と実行時間を集約し，SNR ごとの比較図（学習点数 vs. RMSE）を自動生成します。
- 評価時間を短く抑えたい場合は `--eval_duration` を指定し，テスト区間の先頭から所望の秒数分だけを評価に利用します。

## 処理フロー
1. YAML 設定ファイルを読み込み，対象システム・掃引する `r_list`／`SNR_dB`／乱数種を取得。
2. 各条件について真値軌道の生成と観測サンプリングを行い，学習／テスト分割を作成。
3. 各モデルを順に学習し，指定した評価区間でロールアウト → RMSE と実行時間を記録。
4. 収集したメトリクスを JSON として保存しつつ，SNR ごとに `n_train` をキーとした平均値・標準偏差を集計。
5. `dynid_benchmark.io.viz.plot_model_comparison` を用いて，可視化図（学習点数 vs. RMSE）を出力。

## 主な引数
- `--config`: 既存 YAML（例: `exp/B1_OU_noise.yaml`）。`r_list` がそのまま学習点数掃引の軸になります。
- `--models`: カンマ区切りで比較したいモデル名。`MODEL_REGISTRY` に登録済みであれば利用可能。
- `--time`: シミュレーション全体時間の上書き。短時間での検証向け。
- `--eval_duration`: テスト区間の先頭から評価する秒数。未指定ならテスト区間全体を使用。
- `--outdir`: `runs/<config>/<tag>/` 以下に個別結果を保管。集約図は `runs/<config>/comparison_SNR*.png` として出力されます。

## 出力物
- `data_train.npz` / `data_test.npz`: 再現用データ。
- `metrics_<model>.json`: モデル別メトリクス（学習点数，RMSE，計算時間を含む）。
- `comparison_results.json`: すべての条件をまとめたリスト形式のログ。
- `comparison_SNR<value>.png`: 学習点数と RMSE の関係をモデルごとに比較した図。平均値 ± 標準偏差の帯を描画します。

## 今後の拡張案
- `--train_fractions` など，分割後の訓練系列をさらにサブサンプリングしてデータ量を細かく制御するオプション。
- RMSE 以外の指標（例: フェーズエラー，推定時間）を図示する追加可視化。
- `comparison_results.json` からレポート（Markdown や CSV）を生成する簡易 CLI。
