# プロジェクト実行ガイド (Howto.md)

このドキュメントでは、開発したNIH Chest X-ray病変分類・可視化パイプラインの動作確認方法と、本格的な学習の実行方法について説明します。

## 前提条件

* Docker Desktopがインストールされ、起動していること。
* プロジェクトのルートディレクトリにいること。
* `data/nih-chest-x-ray` にデータセットが配置されていること（詳細は `INSTRUCTIONS.md` や `README.md` を参照）。

## 1. 動作確認 (Smoke Test)

実装したパイプラインが正常に動作するか、少量のデータを使って検証します。

1.  **コンテナの起動**
    ```bash
    docker compose up -d --build
    ```

2.  **検証スクリプトの実行**
    このスクリプトは、データ読み込み、モデル推論、Grad-CAM生成の一連の流れをテストします。GPUがない環境でも動作するように調整されています。
    ```bash
    docker compose exec app python verify_pipeline.py
    ```

3.  **結果の確認**
    コマンドが正常終了すると、プロジェクトのルートディレクトリに `verification_result.png` という画像ファイルが生成されます。これを開いて、X線画像にヒートマップが重畳されていることを確認してください。

## 2. 学習の実行 (Training)

動作確認ができたら、データセット全体を使ってモデルの学習を行います。

### 基本的な学習コマンド

デフォルトの設定（`conf/config.yaml` および `conf/model/resnet50.yaml`）で学習を開始します。

```bash
docker compose exec app python main.py
```

### 設定の変更 (Hydra)

Hydraを使用しているため、コマンドライン引数で設定を柔軟に変更できます。

*   **エポック数の変更**:
    ```bash
    docker compose exec app python main.py epochs=10
    ```

*   **モデルの変更**:
    `conf/model/` にある他のモデル（`densenet121`, `efficientnet` 等）に切り替える場合：
    ```bash
    docker compose exec app python main.py model=densenet121
    ```

*   **バッチサイズの変更**:
    ```bash
    docker compose exec app python main.py dataset.batch_size=64
    ```

*   **複数実験の同時実行 (Multirun)**:
    複数のモデルを一度に比較実験したい場合：
    ```bash
    docker compose exec app python main.py --multirun model=resnet50,densenet121 epochs=5
    ```

### 学習のモニタリング (TensorBoard)

学習の進捗（Loss, Accuracy）はTensorBoardで確認できます。

1.  ブラウザで [http://localhost:6006](http://localhost:6006) にアクセスします。
2.  `outputs/` ディレクトリに出力されたログが可視化されます。

## 3. トラブルシューティング

*   **データが見つからないエラー**: `conf/dataset/nih_chest_x_ray.yaml` の `data_dir` パスが正しいか確認してください。
*   **Grad-CAMのエラー**: `verify_pipeline.py` は成功するが `main.py` で失敗する場合、モデルのレイヤー構造と `conf/model/*.yaml` の `target_layer` の指定が一致していない可能性があります。
