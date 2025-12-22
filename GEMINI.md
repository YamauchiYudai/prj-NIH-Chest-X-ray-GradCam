# Prototype Instruction: VinDR-CXR Classification & Grad-CAM (Docker + Hydra)

## コンテキスト
私は医療画像工学エンジニアです。Docker環境上で、Hugging Faceの `Dangindev/VinDR-CXR-VQA` データセットを使用し、**複数のモデル**を用いた病変分類の比較実験と、Grad-CAMによる可視化を行いたいです。実験管理には **Hydra** を使用します。

## 要件
以下の仕様に基づき、拡張性と実験再現性の高いPythonコードを作成してください。

### 技術スタック
- **Environment**: Docker (Base: `pytorch/pytorch`)
- **Config**: Hydra (`hydra-core`, `omegaconf`)
- **ML**: PyTorch, Torchvision, timm (必要であれば)
- **XAI**: `pytorch-gradcam`
- **Logging**: TensorBoard
- **Data**: Hugging Face Datasets

### 実装ステップ

#### Step 0: Docker環境と依存関係
- `Dockerfile`: `hydra-core`, `timm` (EfficientNet等用), `scikit-learn`, `tensorboard` 等を追加。
- `docker-compose.yml`: データセットのマウントとGPU設定、TensorBoardポートの開放。

#### Step 1: Hydraによる構成管理 (`conf/`)
- 以下の階層構造を持つ設定ファイルを作成してください。
  ```text
  conf/
  ├── config.yaml          # 全体設定 (defaultsリスト含む)
  ├── model/
  │   ├── resnet50.yaml    # target_layer: "layer4"
  │   ├── densenet121.yaml # target_layer: "features"
  │   └── efficientnet.yaml # target_layer: "features" (要確認)
  └── dataset/
      └── vindr.yaml

#### Step 2: データセットと前処理

* `src/data/dataset.py`: Hydraのconfigを受け取り、画像サイズやバッチサイズを動的に設定。
* クラス定義は主要な病変（例: "Aortic enlargement", "Cardiomegaly", "Pleural effusion"）に絞る。

#### Step 3: マルチモデル構築 (`src/models/`)

* Factoryパターン等を用い、Configの `model.name` に応じてモデルを切り替える関数 `get_model(cfg)` を作成してください。
* **ResNet50**: `torchvision`
* **DenseNet121**: `torchvision` (CXRで高精度な傾向あり)
* **EfficientNet-B0**: `torchvision` or `timm`


* 全て `pretrained=True` とし、出力層（Classifier）をデータセットのクラス数に合わせる。

#### Step 4: Grad-CAMの実装 (モデル非依存)

* `src/visualization/gradcam.py`:
* Config内の `model.target_layer` 文字列情報を使って、Grad-CAMの対象層を動的に指定するロジックを実装してください。
* **重要**: モデルによって層の構造が違うため、`getattr` や再帰的な取得が必要になる場合があります。



#### Step 5: メイン実行スクリプト (`main.py`)

* `@hydra.main` デコレータを使用。
* 学習（Fine-tuning）→ 推論 → Grad-CAM生成を一連の流れとして実行。
* 学習の進捗（Loss/Accuracy）はTensorBoardに記録。
* 結果は `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}` に保存されるHydraのデフォルト挙動を利用。

### 出力ファイルの構成

* `conf/` (各yaml)
* `src/data/`, `src/models/`, `src/utils/`
* `main.py`
* `Dockerfile`, `requirements.txt`

## 制約

* **Docker完結**: ホスト環境不使用。
* **可読性**: モデルごとのターゲット層の違いをConfigに外出しし、コード内のハードコーディングを避けること。
* **型ヒント**: 必須。
