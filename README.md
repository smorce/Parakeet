https://github.com/smorce/Parakeet


## 使い方

### 効率的な起動方法（推奨）

開発時は以下の効率的な起動方法を使用してください：

1. **高速起動（推奨）** - 既存のコンテナを再利用
```bash
./dev.sh
```

2. **通常起動** - キャッシュを活用したビルド
```bash
./start.sh
```

3. **クリーンビルド** - 全て再ビルド（依存関係を変更した場合）
```bash
./start.sh --clean
```

4. **依存関係チェック** - どの起動方法が適切かを提案
```bash
./check-deps.sh
```

### 起動モードの使い分け

- **初回起動時**: `./start.sh` で通常ビルド
- **日常の開発**: `./dev.sh` で高速起動（数秒で起動）
- **requirements.txt変更後**: `./start.sh --clean` でクリーンビルド
- **迷った時**: `./check-deps.sh` で適切なモードを確認

終了したあとは
```bash
docker compose down
```


# リアルタイム文字起こしアプリケーション

## 概要

このプロジェクトは、マイクからの音声入力をリアルタイムで文字起こしし、Webインターフェースに表示するアプリケーションです。Dockerを使用して環境を構築し、NVIDIA NeMoのASRモデル、Silero VADによる音声区間検出、GradioによるUIを組み合わせています。

## 主な特徴

- **リアルタイム文字起こし:** マイクからの音声を低遅延でテキストに変換します。
- **音声区間検出 (VAD):** Silero VADを用いて無音区間を除外し、効率的に発話のみを処理します。
- **Webインターフェース:** Gradioを利用したシンプルで直感的なUIを提供します。「開始」「停止」ボタンで簡単に操作できます。
- **コンテナ化:** DockerとDocker Composeにより、依存関係を含めた実行環境を簡単に構築・再現できます。
- **GPU対応:** NVIDIA GPUを自動で検出し、高速な文字起こし処理を実現します。GPUが利用できない場合はCPUにフォールバックします。

## アーキテクチャ

本システムは、以下のコンポーネントがマルチスレッドで並列動作することで実現されています。

- **Gradio UI (メインスレッド):** ユーザー操作の受付と結果のストリーミング表示を担当します。
- **音声入力スレッド:** `sounddevice` を用いてマイクから音声を取得します。
- **VAD処理スレッド:** `silero-vad` を用いて音声区間を検出します。
- **文字起こしスレッド:** NVIDIA NeMo ASRモデル (`nvidia/parakeet-tdt_ctc-0.6b-ja`) で文字起こしを実行します。

```mermaid
graph TD
    subgraph "Gradio App (Main Thread)"
        A[Gradio UI <br> (Web Interface)] -- ボタン操作 --> B{録音状態フラグ};
        A -- "yield"による更新 --> C[出力用テキストボックス];
        D[結果キュー<br>(queue.Queue)] -- UI更新ループが読み出し --> A;
    end

    subgraph "ワーカースレッド1: 音声入力"
        E[マイク<br>(sounddevice)] -- 音声チャンク --> F[音声キュー<br>(queue.Queue)];
    end
    B -- 状態を監視 --> E;

    subgraph "ワーカースレッド2: VAD"
        G[VAD処理<br>(silero-vad)] -- get --> F;
        G -- 発話音声 --> H[文字起こしキュー<br>(queue.Queue)];
    end

    subgraph "ワーカースレッド3: 文字起こし"
        I[Nemo ASRモデル<br>(Parakeet)] -- get --> H;
        I -- テキスト --> D;
    end
```

## 実行方法

### 前提条件

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/) (Docker Desktopには通常含まれています)
- ホストマシンに接続されたマイク

### 手順

1.  このリポジトリをクローンまたはダウンロードします。

2.  ターミナルでプロジェクトのルートディレクトリに移動します。

3.  以下のコマンドを実行して、Dockerイメージをビルドし、コンテナを起動します。
    ```bash
    docker compose up --build
    ```
    初回起動時はモデルのダウンロードなどにより時間がかかる場合があります。

4.  ビルドと起動が完了したら、Webブラウザで以下のURLにアクセスします。
    [http://localhost:3791](http://localhost:3791)

5.  表示されたGradioのインターフェースで「録音開始」ボタンを押すと、マイクからの音声入力が始まり、リアルタイムで文字起こし結果が表示されます。

## ファイル構成

-   `Dockerfile`: アプリケーションの実行環境を定義します。CUDA、Python、必要なシステムライブラリ（`portaudio19-dev`など）をインストールします。
-   `compose.yaml`: Docker Composeの設定ファイルです。サービスのビルド方法、ポートマッピング(`3791:3791`)、マイクデバイスのマウントなどを定義します。
-   `requirements.txt`: Pythonの依存ライブラリをリストします。
-   `scripts/realtime_transcribe_gradio.py`: アプリケーションの本体です。Gradio UI、音声処理、文字起こしのロジックが実装されています。

## カスタマイズ

### VAD感度の調整

音声検出の感度は、環境ノイズや話者の声量に合わせて調整できます。
`scripts/realtime_transcribe_gradio.py` 内の以下の行の `threshold` 値を変更してください。

```python
# scripts/realtime_transcribe_gradio.py

# ...
vad_iterator = VADIterator(vad_model, threshold=0.7) # この値を調整 (0.0 ~ 1.0)
# ...
```

-   `threshold` を高くすると、より大きな音でないと発話として認識されにくくなります（ノイズが多い環境向け）。
-   `threshold` を低くすると、小さな音でも発話として認識されやすくなります（静かな環境向け）。
