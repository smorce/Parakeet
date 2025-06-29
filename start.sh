#!/bin/bash

# realtime speech to text 起動スクリプト
# 既存のコンテナ・イメージを削除してからクリーンビルド・起動

set -e  # エラー時に終了

echo "🚀 realtime speech to text を起動します..."

# .envファイルの存在確認
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

# 関連するコンテナを強制削除（念のため）
echo "🗑️  関連コンテナを強制削除中..."
docker rm -f realtime-speech-to-text-container 2>/dev/null || true

# 既存のイメージを削除
echo "🗑️  既存のイメージを削除中..."
docker rmi -f realtime-speech-to-text:latest 2>/dev/null || true

# クリーンビルド・起動
echo "🔨 イメージをビルド中..."
docker compose build --no-cache

echo "▶️  サービスを起動中..."
# echo "📍 アプリケーションは http://localhost:3791/ でアクセス可能です"
echo "⏹️  停止するには Ctrl+C を押してください"
docker compose up

echo "🛑 サービスを停止中..."
docker compose down
echo "🛑 サービスを停止完了！" 