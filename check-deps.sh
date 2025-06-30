#!/bin/bash

# 依存関係変更検知スクリプト
# requirements.txtが変更された場合、適切なビルドモードを提案

set -e

echo "🔍 依存関係の変更をチェック中..."

# Dockerイメージが存在するかチェック
if ! docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "realtime-speech-to-text:latest"; then
    echo "📦 Dockerイメージが存在しません。初回ビルドが必要です。"
    echo "💡 実行推奨: ./start.sh"
    exit 0
fi

# requirements.txtのタイムスタンプとイメージ作成日時を比較
REQ_TIMESTAMP=$(stat -c %Y requirements.txt 2>/dev/null || echo 0)
IMG_TIMESTAMP=$(docker inspect --format='{{.Created}}' realtime-speech-to-text:latest 2>/dev/null | xargs -I {} date -d {} +%s || echo 0)

if [ "$REQ_TIMESTAMP" -gt "$IMG_TIMESTAMP" ]; then
    echo "⚠️  requirements.txtがDockerイメージより新しいです！"
    echo "🔄 依存関係の変更が検出されました。リビルドが推奨されます。"
    echo ""
    echo "推奨アクション:"
    echo "  クリーンビルド: ./start.sh --clean"
    echo "  または通常ビルド: ./start.sh"
    exit 1
else
    echo "✅ 依存関係に変更はありません。高速起動が可能です。"
    echo "💡 実行推奨: ./dev.sh"
    exit 0
fi 