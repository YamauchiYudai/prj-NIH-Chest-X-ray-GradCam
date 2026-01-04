# NIH Chest X-ray Classification & Grad-CAM Analysis

## 概要 (Overview)
本プロジェクトは、NIH Chest X-ray データセットを用いて胸部X線画像の病変分類（マルチラベル分類）を行い、Grad-CAM (Gradient-weighted Class Activation Mapping) を用いてモデルの判断根拠を可視化するアプリケーションです。

Docker環境で完結し、Hydraを用いた柔軟な実験構成管理、TensorBoardによる学習モニタリングをサポートしています。

## 特徴 (Features)
*   **マルチモデル対応**: ResNet50, DenseNet121, EfficientNetなどを設定ファイルで切り替え可能。
*   **Grad-CAM 可視化**: 学習済みモデルが画像のどこに注目しているかをヒートマップで表示。
*   **実験管理 (Hydra)**: パラメータ（エポック数、学習率、モデル種別など）をコマンドラインから簡単に変更・管理。
*   **再現性**: Dockerコンテナにより、環境差異を排除した実行が可能。

## ディレクトリ構造

```
prj-VinDRCXR-GradCam/
├── conf/                   # Hydra設定ファイル (config.yaml, model/*.yaml, dataset/*.yaml)
├── data/                   # データセット格納場所 (NIH Chest X-ray images & CSV)
├── outputs/                # 学習結果の出力先 (ログ、モデル重み、Grad-CAM画像)
├── src/                    # ソースコード
│   ├── data/               # データ読み込み・前処理
│   ├── models/             # モデル定義
│   ├── utils/              # 学習ループ、パス解決などのユーティリティ
│   └── visualization/      # Grad-CAM実装
├── Dockerfile              # Docker環境定義
├── docker-compose.yml      # コンテナ構成
├── main.py                 # 学習・評価・可視化のメインスクリプト
├── verify_pipeline.py      # 動作確認用スクリプト (Smoke Test)
└── requirements.txt        # Python依存ライブラリ
```

## 前提条件 (Prerequisites)
*   **Docker Desktop**: インストール済みであること。
*   **データセット**: NIH Chest X-ray Dataset を `data/` ディレクトリに配置済みであること。
    *   `data/images_001` 〜 `data/images_012` および `train_val_list.txt`, `test_list.txt` 等が存在する状態。

## 使い方 (Usage)

### 1. 環境構築と起動
プロジェクトのルートディレクトリで以下のコマンドを実行し、Dockerコンテナをビルド・起動します。

```bash
docker compose build
docker compose up -d
```

### 2. 開発用テスト (Development Test)
開発用の統合テスト（Smoke Test）を実行します。このテストはダミーデータを `src/test/test_data` に自動生成し、学習パイプライン全体がエラーなく動作することを確認します。GitHub ActionsのCIでも実行されます。

```bash
docker compose run --rm app python -m src.test.test_main
```

### 3. 動作確認 (Legacy Smoke Test)
少量のデータを使ってパイプライン全体（読み込み→推論→Grad-CAM）が正常に動くか確認します。
GPUがない環境でも動作します。

```bash
docker compose exec app python verify_pipeline.py
```
*   成功すると `Verification Completed` と表示され、ルートディレクトリに `verification_result.png` が生成されます。

### 4. モデルの学習 (Training)
データセット全体を使って学習を実行します。デフォルトでは `resnet50` を使用し、テストセットでの評価とGrad-CAM生成まで行います。

```bash
docker compose exec app python main.py
```

#### 4.1. 学習の高速化 (Optional)
画像の読み込み（I/O）時間を短縮するため、事前にデータをPickle化してメモリに展開してから学習することができます。

1.  **前処理（初回のみ）**: 画像をリサイズしてPickleファイルに変換します。
    ```bash
    docker compose run --rm app python -m src.preprocessing.create_picklefiles
    ```
2.  **高速学習の実行**: `dataset.use_pickle=true` を指定します。
    ```bash
    docker compose exec app python main.py dataset.use_pickle=true
    ```

#### パラメータの変更
Hydraの機能により、コマンドライン引数で設定を上書きできます。

*   **エポック数を変更**:
    ```bash
    docker compose exec app python main.py epochs=10
    ```
*   **モデルを変更** (`densenet121`):
    ```bash
    docker compose exec app python main.py model=densenet121
    ```
*   **バッチサイズを変更**:
    ```bash
    docker compose exec app python main.py dataset.batch_size=64
    ```

### 4. 学習状況のモニタリング (TensorBoard)
学習中のLossやAccuracyの推移をブラウザで確認できます。

1.  ブラウザで [http://localhost:6006](http://localhost:6006) にアクセス。
2.  `outputs/` 以下のログが自動的に表示されます。

### 5. 推論と可視化 (Inference & Grad-CAM)
`main.py` の実行完了時、以下の処理が自動で行われます。

1.  **テストセット評価**: テストデータのAccuracy（Binary Accuracy）を表示。
2.  **Grad-CAM生成**:
    *   `gradcam_grid_{model_name}.png`: テスト画像からランダムに生成した可視化結果一覧。
    *   `gradcam_{Pathology}.png`: 各病変クラスごとの代表的な可視化結果。

これらは実行時の出力ディレクトリ（`outputs/YYYY-MM-DD/HH-MM-SS/`）内に保存されます。TensorBoardのログや `best_model.pth` も同ディレクトリ内に保存されます。

---

## 補足: 単一画像の推論
(開発中機能) `predict.py` を使用して、特定の画像ファイルに対する推論を行うことも可能です。

```bash
# 例: 学習済みモデルのパスと設定フォルダを指定して実行
docker compose exec app python predict.py outputs/YYYY-MM-DD/HH-MM-SS/best_model.pth data/images_001/images/00000001_000.png --config_path outputs/YYYY-MM-DD/HH-MM-SS/
```
