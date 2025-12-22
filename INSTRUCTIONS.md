# プロジェクト実行ガイド

このドキュメントは、プロジェクトの環境構築から学習、推論、結果の可視化までの完全な手順を説明します。

## 1. 前提条件

-   **Docker と Docker Compose** がインストールされていること。
    -   [Docker Desktop](https://www.docker.com/products/docker-desktop/) のインストールを推奨します。

## 2. データセットの準備

このプロジェクトは、手動でダウンロードする必要がある医療画像データセット `VinDr-CXR` を使用します。

1.  **PhysioNetからデータセットをダウンロード:**
    -   [VinDr-CXR (version 1.0.0)](https://physionet.org/content/vindr-cxr/1.0.0/) にアクセスします。
    -   `files` セクションから `vindr-cxr-1.0.0.zip` をダウンロードします（サイズが大きいため時間がかかります）。

2.  **データセットを解凍:**
    -   ダウンロードした `vindr-cxr-1.0.0.zip` を解凍します。

3.  **プロジェクトフォルダに配置:**
    -   このプロジェクトのルート (`prj-VinDRCXR-GradCam/`) に `data` という名前のディレクトリを作成します。
    -   `data` ディレクトリの中に、解凍したフォルダ一式（`1.0.0` というディレクトリなど）を移動します。
    -   最終的なディレクトリ構造は以下のようになります。
        ```
        prj-VinDRCXR-GradCam/
        └── data/
            └── physionet.org/
                └── files/
                    └── vindr-cxr/
                        └── 1.0.0/
                            ├── annotations/
                            ├── core/
                            ├── train/
                            └── test/
        ```
    -   設定ファイル (`conf/dataset/vindr.yaml`) の `data_dir` は、この `1.0.0` ディレクトリを指すようにデフォルトで設定されています。

## 3. Docker環境の構築

ターミナルを開き、プロジェクトのルートディレクトリで以下のコマンドを実行します。

1.  **Dockerイメージをビルド:**
    -   必要なPythonライブラリをインストールし、環境を構築します。
    ```bash
    docker compose build
    ```

2.  **コンテナをバックグラウンドで起動:**
    ```bash
    docker compose up -d
    ```

## 4. モデルの学習

学習は `docker compose exec` コマンドを介して実行します。

### A) 単一実験の実行

特定のモデル（例: `resnet50`）で、エポック数を指定して学習を実行します。

```bash
docker compose exec app python main.py model=resnet50 epochs=5
```

### B) 複数実験の比較実行（推奨）

Hydraの強力な機能である `multirun` を使うと、複数のモデルやパラメータでの実験を一度に実行できます。結果は個別のディレクトリに出力され、TensorBoardでの比較が容易になります。

**例: ResNet50とDenseNet121の両方を学習させる**
```bash
docker compose exec app python main.py --multirun model=resnet50,densenet121 epochs=5
```

## 5. 学習結果の比較 (TensorBoard)

`multirun` で実行した実験結果は、TensorBoardで視覚的に比較するのが最も効率的です。

1.  **TensorBoardを起動:**
    -   学習の実行とは **別のターミナル** を開き、以下のコマンドを実行します。
    -   `--logdir outputs` は、Hydraが出力したすべての実験結果 (`outputs` ディレクトリ以下）を監視対象とすることを意味します。
    ```bash
    docker compose exec app tensorboard --logdir outputs
    ```

2.  **ブラウザで確認:**
    -   ブラウザで `http://localhost:6006` を開きます。
    -   TensorBoardの画面で、各モデルの学習曲線（AccuracyやLoss）を重ねて表示したり、ハイパーパラメータによる結果の違いを分析したりできます。

## 6. 学習済みモデルによる推論

`predict.py` スクリプトを使用して、学習済みのモデルで新しい画像の分類推論を実行できます。

-   **推論の実行:**
    ```bash
    docker compose exec app python predict.py [モデルのパス] [画像のパス] --config_path [Hydra設定のパス]
    ```

-   **引数の説明:**
    -   `[モデルのパス]`: 学習時に保存された `.pth` ファイルのパス。例: `outputs/2023-12-22/10-00-00/best_model.pth`
    -   `[画像のパス]`: 推論したいDICOMまたはPNG/JPEG画像のパス。
    -   `[Hydra設定のパス]`: モデルを学習させた際のHydraの出力ディレクトリ。モデルの構造を正しく復元するために必要です。例: `outputs/2023-12-22/10-00-00/`

-   **実行例:**
    ```bash
    docker compose exec app python predict.py outputs/multirun/2025-12-22/22-50-10/0/best_model.pth data/physionet.org/files/vindr-cxr/1.0.0/test/00a85be6-6eb0-421d-8acf-ff2dc0007e8a.dicom --config_path outputs/multirun/2025-12-22/22-50-10/0/
    ```

## 7. Grad-CAMによる結果の解釈

学習が完了すると、モデルが画像のどの部分に注目して判断したかを可視化する **Grad-CAM** 画像が自動的に生成されます。

-   **ランダムなテスト画像での可視化:**
    -   `gradcam_grid_resnet50.png` のような名前で、テストセットからランダムに選ばれた画像に対する可視化結果が保存されます。

-   **病変ごとの可視化:**
    -   `gradcam_Aortic_enlargement.png` のように、`conf/dataset/vindr.yaml` で定義された各病変について、代表的な画像の可視化結果が1枚ずつ保存されます。これにより、特定の病変に対してモデルがどこを重要視しているかを個別に確認できます。
