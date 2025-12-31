# Release Notes

## v1.0.0 (2025-12-28)

**NIH Chest X-ray Classification & Grad-CAM Pipeline** の初回リリースです。
Docker環境上で動作する、胸部X線画像の病変分類および判断根拠の可視化（Grad-CAM）システムを提供します。

### ✨ 主な機能 (Key Features)

*   **Docker完全対応**: ホスト環境を汚さず、`docker compose up` だけで再現性のある実験環境を構築できます。
*   **マルチモデル・アーキテクチャ**: 設定ファイル (`conf/`) を変更するだけで、以下のモデルを切り替えて実験可能です。
    *   ResNet50
    *   DenseNet121
    *   EfficientNet
*   **Explainable AI (XAI)**: Grad-CAM (Gradient-weighted Class Activation Mapping) を実装。モデルが画像のどの領域に注目して病変を検出したかをヒートマップで可視化します。
*   **高度な実験管理**: Hydra + OmegaConf を採用し、学習率、エポック数、モデルパラメータなどをコマンドライン引数で柔軟に変更・管理できます。
*   **学習モニタリング**: TensorBoardにより、LossやAccuracyの推移をリアルタイムで可視化可能です。
*   **検証用スクリプト**: `verify_pipeline.py` を同梱しており、少量のデータを用いたパイプラインの動作確認（スモークテスト）が容易に行えます。

### 🚀 クイックスタート (Quick Start)

1.  **データセットの準備**:
    NIH Chest X-ray データセットを `data/` ディレクトリに配置してください。

2.  **環境の起動**:
    ```bash
    docker compose up -d --build
    ```

3.  **動作確認 (Smoke Test)**:
    ```bash
    docker compose exec app python verify_pipeline.py
    ```

4.  **学習の実行**:
    ```bash
    docker compose exec app python main.py
    ```

### 🛠 技術スタック (Tech Stack)

*   **Platform**: Docker, Docker Compose
*   **ML Framework**: PyTorch, Torchvision, Timm
*   **Config Management**: Hydra, OmegaConf
*   **Visualization**: Grad-CAM, TensorBoard
*   **Data Processing**: Pandas, NumPy, Scikit-learn

### 📝 既知の問題 / 注意事項

*   学習には十分なメモリリソースが必要です。
*   `data/` ディレクトリ以下の画像配置構造は、`src/utils/file_finder.py` のロジックに従う必要があります（README参照）。
