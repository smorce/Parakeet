#!/bin/bash

# realtime speech to text 起動スクリプト
# 効率的なコンテナ管理でライブラリの再インストールを最小化

set -e  # エラー時に終了

echo "🚀 realtime speech to text を起動します..."

# 起動モードの選択
if [ "$1" = "--clean" ] || [ "$1" = "-c" ]; then
    CLEAN_BUILD=true
    echo "🧹 クリーンビルドモードで実行します（全て再ビルド）"
else
    CLEAN_BUILD=false
    echo "⚡ 高速モードで実行します（キャッシュ活用）"
    echo "💡 完全な再ビルドが必要な場合は './start.sh --clean' を実行してください"
fi

# .envファイルの存在確認（コメントアウト状態を維持）
# if [ ! -f ".env" ]; then
#     echo "❌ エラー: .envファイルが見つかりません"
#     echo "   以下のコマンドで.envファイルを作成してください:"
#     echo "   echo 'GEMINI_API_KEY=your_gemini_api_key_here' > .env"
#     exit 1
# fi

# echo "📋 .envファイルが確認できました"

# 既存のコンテナを停止・削除
echo "🛑 既存のコンテナを停止・削除中..."
docker compose down --remove-orphans 2>/dev/null || true

if [ "$CLEAN_BUILD" = true ]; then
    # クリーンビルドモード: コンテナとイメージを完全削除
    echo "🗑️  関連コンテナを強制削除中..."
    docker rm -f realtime-speech-to-text-container 2>/dev/null || true
    
    echo "🗑️  既存のイメージを削除中..."
    docker rmi -f realtime-speech-to-text:latest 2>/dev/null || true
    
    echo "🔨 イメージをクリーンビルド中（時間がかかります）..."
    docker compose build --no-cache
else
    # 高速モード: キャッシュを活用
    echo "🔨 イメージをビルド中（キャッシュ活用）..."
    docker compose build
fi

echo "▶️  サービスを起動中..."
echo "📍 アプリケーションは http://localhost:3791/ でアクセス可能です"
echo "⏹️  停止するには Ctrl+C を押してください"
docker compose up

echo "🛑 サービスを停止中..."
docker compose down
echo "🛑 サービスを停止完了！" 