やっぱり、Docker＋WSLでのマイク接続が無理そう。

import sys, os
import numpy as np

from fastrtc import ReplyOnPause
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.stream import Stream

"""
マイクの音声をそのままスピーカーにエコーバックするだけのサンプル
"""
def echoback(audio: tuple[int, np.ndarray]):
    print(f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]}秒の音声が入力されました。")
    yield audio

def example_echoback():
    algo_options = AlgoOptions(
        audio_chunk_duration=0.6,
        started_talking_threshold=0.5,
        speech_threshold=0.1,
    )
    stream = Stream(
        handler=ReplyOnPause(
            echoback,
            algo_options=algo_options,
            input_sample_rate=16000,
            output_sample_rate=16000,
        ),
        modality="audio", 
        mode="send-receive",
    )

    # CORS設定を追加してWebRTC接続を改善
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="FastRTC Simple Echo Server",
        description="音声をそのままエコーバックするサーバー",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    stream.mount(app)
    stream.ui.launch(server_port=6075, server_name="0.0.0.0")

if __name__ == "__main__":
    example_echoback()