# CI/CD パイプライン (GitHub Actions)

このプロジェクトには、GitHub Actionsを用いた自動テストとデプロイの仕組み（CI/CD）が導入されています。

## ワークフローの概要

設定ファイル: `.github/workflows/ci-cd.yaml`

### 1. Test (CI)
コードが変更されるたびに実行されるテストフェーズです。

*   **トリガー**: `main` ブランチへのプッシュ、またはプルリクエスト。
*   **内容**:
    1.  Dockerイメージをビルドします。
    2.  CI環境用にダミーのデータリスト (`train_val_list.txt`) を生成します。
    3.  `verify_pipeline.py` を実行し、データセットがない環境でもプログラムがクラッシュせずに動作するか（Smoke Test）を確認します。

### 2. Build & Push (CD)
テストを通過した安定版をDockerイメージとして公開するフェーズです。

*   **トリガー**: `main` ブランチへのプッシュ、または `v*` タグのプッシュ（プルリクエストでは実行されません）。
*   **保存先**: GitHub Container Registry (GHCR)。
*   **内容**:
    1.  Dockerイメージをビルドします。
    2.  `ghcr.io/<ユーザー名>/<リポジトリ名>:tag` としてイメージをプッシュします。

## GitHubの設定

このワークフローを正しく動作させるために、リポジトリの設定を確認してください。

1.  **Actionsの有効化**:
    リポジトリの "Settings" -> "Actions" -> "General" で "Allow all actions and reusable workflows" が選択されていることを確認。

2.  **Packageの権限**:
    イメージをGHCRにプッシュするため、GitHub Actionsに書き込み権限が必要です。
    `.github/workflows/ci-cd.yaml` 内の以下の記述により自動的に付与されます：
    ```yaml
    permissions:
      contents: read
      packages: write
    ```

## イメージの利用方法

ビルドされたイメージは、以下のコマンドでプルできます（リポジトリがPublicの場合、または認証済みの場合）。

```bash
docker pull ghcr.io/<GitHubユーザー名>/vindr-cxr-gradcam:main
```
