https://github.com/smorce/Parakeet


## 使い方
Docker を使ってクリーンアップ・ビルド・起動する
```
./start.sh
```

終了したあとは
docker compose down
する。


docker build -t realtime-speech-to-text:latest .



docker run --gpus all \
  --rm \
  --name realtime-speech-to-text-container \
  -v $(pwd)/audio:/app/audio \
  realtime-speech-to-text:latest \
  /app/audio/sample.wav



