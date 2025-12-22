# Agents Guide (Docker + Hydra Edition)

## TL;DR
- **環境**: Dockerコンテナ内 (`docker compose up` -> `docker compose exec`)。
- **設定**: 全て **Hydra (`conf/`)** で管理。ハードコーディング禁止。
- **実行**: `python main.py model=resnet50` のようにCLI引数で条件を変更。
- **品質**: `ruff` -> `mypy` -> `pytest`。

## 設計原則: Hydraによる実験管理
- **Config Driven**: 学習率、エポック数、画像サイズ、モデルの種類、Grad-CAMの対象層など、変更しうる値は全て `yaml` に定義する。
- **Output Isolation**: 実験結果（ログ、重み、可視化画像）は Hydra が自動生成する `outputs/` ディレクトリ配下に保存し、過去の実験と混ざらないようにする。
- **Model Switching**: `model` グループを切り替えるだけで、アーキテクチャと付随するメタデータ（Grad-CAMのターゲット層など）がセットで切り替わるように設計する。

## ディレクトリ構造
```text
.
├── conf/               # Hydra設定ファイル群
│   ├── model/          # モデルごとのパラメータ
│   └── config.yaml     # エントリーポイント
├── src/
│   ├── data/           # データセット定義
│   ├── models/         # モデル定義 (Factory)
│   └── visualization/  # Grad-CAMなど
├── outputs/            # 実験結果 (git ignore)
└── main.py             # @hydra.main

```

## 実用スニペット

### Hydra Config アクセスとPydantic検証

HydraのDictConfigをそのまま使うのではなく、Pydantic等でバリデーションすることを推奨（今回はプロトタイプなのでDictConfig直利用も可だが、型安全を意識する）。

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Model: {cfg.model.name}")
    print(f"Target Layer: {cfg.model.target_layer}")
    
    # モデル構築
    model = get_model(cfg.model.name, num_classes=cfg.dataset.num_classes)
    
    # ... Training / Inference ...

```

### Grad-CAMのターゲット層 動的取得

文字列から実際のモジュールオブジェクトを取得するヘルパー関数。

```python
def get_target_layer(model, layer_name: str):
    # 例: "layer4.2.conv2" のようなドット区切りに対応
    modules = layer_name.split(".")
    target = model
    for m in modules:
        target = getattr(target, m)
    return [target]

```

## ワークフロー (Docker)

1. **比較実験の実行**:
```bash
# ResNet50で実験
python main.py model=resnet50 dataset.batch_size=32

# DenseNet121で実験
python main.py model=densenet121 dataset.batch_size=16

```


2. **結果の確認**:
`outputs/YYYY-MM-DD/HH-MM-SS/` 内に保存された `gradcam_*.png` を確認する。

3. **TensorBoardの起動**:
別のターミナルで以下のコマンドを実行し、学習結果を比較する。
```bash
# Dockerコンテナ内でtensorboardを起動
docker compose exec app tensorboard --logdir outputs/

# ホストのブラウザで http://localhost:6006 を開く
```

## テスト要件

* Configが正しく読み込めるかテストする (`hydra.initialize` を使用)。
* 各モデル（ResNet, DenseNet等）が指定された次元で出力されるかテストする。

