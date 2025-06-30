#!/bin/bash

# realtime speech to text 開発用高速起動スクリプト
# 既存のコンテナを再利用し、最高速での起動を実現

set -e  # エラー時に終了

echo "⚡ realtime speech to text を高速モードで起動します..."

# 既存のコンテナが動いているかチェック
if docker ps -q -f name=realtime-speech-to-text-container | grep -q .; then
    echo "🔄 既存のコンテナが動作中です。再起動します..."
    docker restart realtime-speech-to-text-container
    echo "✅ コンテナを再起動しました！"
    echo "📍 アプリケーションは http://localhost:3791/ でアクセス可能です"
    exit 0
fi

# 停止中のコンテナがあるかチェック
if docker ps -a -q -f name=realtime-speech-to-text-container | grep -q .; then
    echo "▶️  既存のコンテナを開始します..."
    docker start realtime-speech-to-text-container
    echo "✅ コンテナを開始しました！"
    echo "📍 アプリケーションは http://localhost:3791/ でアクセス可能です"
    exit 0
fi

# コンテナが存在しない場合は通常のビルドから起動
echo "🔨 初回起動: イメージをビルドします..."
docker compose build

echo "▶️  サービスを起動中..."
echo "📍 アプリケーションは http://localhost:3791/ でアクセス可能です"
echo "⏹️  停止するには Ctrl+C を押してください"

# バックグラウンドで起動して、ログを表示
docker compose up -d
docker logs -f realtime-speech-to-text-container 