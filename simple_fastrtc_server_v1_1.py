やっぱり、Docker＋WSLでのマイク接続が無理そう。

import sys, os
import numpy as np
import json

from fastrtc_jp.speech_to_text.sr_google import GoogleSTT
# from fastrtc_jp.speech_to_text.vosk import VoskSTT
# from fastrtc_jp.text_to_speech.voicevox import VoicevoxTTSModel, VoicevoxTTSOptions
# from fastrtc_jp.text_to_speech.gtts import GTTSModel, GTTSOptions


"""
マイクの音声をSTT->TTSしてエコーバックするサンプル
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from fastrtc import (
    Stream,
    ReplyOnPause,
    AlgoOptions,
    SileroVadOptions,
    AdditionalOutputs
)

from fastrtc_jp.text_to_speech.style_bert_vits2 import (
    StyleBertVits2,
    StyleBertVits2Options
)
from dataclasses import dataclass
# 元のファイルで必要なインポートが既にあることを前提とします
from style_bert_vits2.constants import DEFAULT_BERT_TOKENIZER_PATHS, Languages, DEFAULT_STYLE
from fastrtc_jp.text_to_speech.opt import SpkOptions
# from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel as SBV2_TTSModel
from numpy.typing import NDArray
# import asyncio

# --- オーバーライドして infer を細かく指定できるようにした ---

@dataclass
class CustomStyleBertVits2Options(StyleBertVits2Options):
    """
    StyleBertVits2Optionsを拡張し、追加の推論パラメータを定義します。
        sdp_ratio: テンポの緩急を調整します。
        length: 話速（スピード）を調整します。
        noise: 音声のランダム性（声の震えなど）を調整します。
        pitch_scale: ピッチ（声の高さ）を調整します。
    """
    sdp_ratio: float | None = None          # DPとSDPの混合比 (0.0 ~ 1.0)
    length_scale: float | None = None       # 話速 (1.0が基準)
    noise_scale: float | None = None        # ノイズの大きさ (0.0 ~ 1.0程度)
    pitch_scale: float | None = None        # ピッチの高さ (1.0が基準)
    # 他のパラメータ（noise_w, intonation_scaleなど）も同様に追加できます。


class CustomStyleBertVits2(StyleBertVits2):
    """
    StyleBertVits2を継承し、推論時の引数をカスタマイズ可能にするクラス。
    _runメソッドをオーバーライドして、inferに渡す引数を拡張します。
    """
    async def _run(self, text:str, options:CustomStyleBertVits2Options|None=None) -> tuple[int, NDArray[np.float32]]:
        # 親クラスの_loadメソッドを呼び出してモデルを準備
        model:SBV2_TTSModel = await self._load(options)

        # 親クラスと同様に、スピーカーやスタイルの基本設定を行う
        speaker_id:int = 0 if 0 in model.id2spk else list(model.id2spk.keys())[0]
        speaker_style:str = DEFAULT_STYLE if DEFAULT_STYLE in model.style2id else list(model.style2id.keys())[0]

        if options:
            if options.speaker_id in model.id2spk:
                speaker_id = options.speaker_id
            elif options.speaker_name in model.spk2id:
                speaker_id = model.spk2id[options.speaker_name]

            if options.speaker_style in model.style2id:
                speaker_style = options.speaker_style

        # model.inferに渡すキーワード引数を動的に構築します
        infer_kwargs = {
            "speaker_id": speaker_id,
            "style": speaker_style,
            # assist_text関連は親クラスの挙動を維持
            "assist_text": self._assist_text,
            "use_assist_text": True
        }

        # カスタムオプションが存在し、値が設定されていればkwargsに追加します
        if options:
            # sdp_ratio
            if options.sdp_ratio is not None:
                infer_kwargs["sdp_ratio"] = options.sdp_ratio
            # length (話速)
            if options.length_scale is not None:
                infer_kwargs["length"] = options.length_scale
            # noise (ノイズ)
            if options.noise_scale is not None:
                infer_kwargs["noise"] = options.noise_scale
            # pitch_scale (ピッチ)
            if options.pitch_scale is not None:
                infer_kwargs["pitch_scale"] = options.pitch_scale

        # 構築した引数を使って推論を実行します
        # ★★★ ここで動的に構築した引数が渡されます ★★★
        frame = model.infer(text, **infer_kwargs)

        # 親クラスと同様にassist_textを更新
        self._assist_text = (self._assist_text + " " + text)[-200:]
        return frame

    # tts, stream_tts, stream_tts_syncメソッドは親クラスのものをそのまま使うため、
    # ここでオーバーライドする必要はありません。


# --- モデル初期化 ---
stt_model = GoogleSTT()
tts_model = CustomStyleBertVits2()
# style_opt = CustomStyleBertVits2Options(
#     device="cuda",
#     model_path="path/to/your/model.safetensors",
#     config_path="path/to/your/config.json",
#     style_vec_path="path/to/your/style_vectors.npy",
#     speaker_style="上機嫌",
#     # ★ここからが追加したカスタムパラメータ★
#     sdp_ratio=0.6,         # テンポに緩急をつける
#     # length_scale=1.2,      # 少しゆっくり話す (1.0より大きく)
#     # noise_scale=0.6,       # ノイズを少し加える
#     # pitch_scale=1.1        # 少し高い声にする (1.0より高く)
# )
style_opt = CustomStyleBertVits2Options(
    device="cuda",
    model="sakura-miko",
    # ★ここからが追加したカスタムパラメータ★
    sdp_ratio=0.6,         # テンポに緩急をつける
    # length_scale=1.2,      # 少しゆっくり話す (1.0より大きく)
    # noise_scale=0.6,       # ノイズを少し加える
    # pitch_scale=1.1        # 少し高い声にする (1.0より高く)
)

# --- 音声ハンドラ ---
def echoback_test(audio: tuple[int, np.ndarray]):
    # 1) 音声認識
    user_text = stt_model.stt(audio)
    print(f"認識結果: {user_text}")  # ログ出力

    # 2) 文字起こしテキストを AdditionalOutputs で返却
    yield AdditionalOutputs(user_text)  # メタデータとして返却　★ここはサンプルコードにはない

    # 3) Style-Bert-VITS2 で音声合成しつつストリーミング返却
    for chunk in tts_model.stream_tts_sync(user_text, style_opt):
        print("Sending audio")
        yield chunk

# --- Stream 作成と VAD 設定 ---
stream = Stream(
    handler=ReplyOnPause(
        echoback_test,
        can_interrupt=True,  # 応答中の割り込みを許可
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,       # チャンク長（秒）
            started_talking_threshold=0.2,  # 発話開始閾値（秒）
            speech_threshold=0.1            # 無音判定閾値（秒）
        ),
        model_options=SileroVadOptions(
            threshold=0.5,                  # 音声検知閾値 (0.0–1.0)
            min_speech_duration_ms=250,     # 最小音声持続時間 (ms)
            min_silence_duration_ms=100     # 無音持続時間 (ms)
        ),
        input_sample_rate=16000,    # サンプルレートを明示的に指定
        output_sample_rate=16000,   # サンプルレートを明示的に指定
    ),
    modality="audio",
    mode="send-receive"
)

# --- FastAPI アプリにマウント ---
app = FastAPI(
    title="FastRTC Echo Server",
    description="音声をSTT->TTSでエコーバックするサーバー",
    version="1.1.0"
)

# CORS設定を追加してWebRTC接続を改善
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

stream.mount(app)  # /webrtc/offer 等のエンドポイントを自動追加

# --- SSE エンドポイント：文字起こし結果配信 ---
@app.get("/test_transcribe")
async def test_transcribe(webrtc_id: str):
    async def event_generator():
        try:
            async for output in stream.output_stream(webrtc_id):
                # output.args[0] が文字起こしテキスト
                yield f"data: {output.args[0]}\n\n"
        except Exception as e:
            print(f"SSE stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# WebRTCエラーハンドリング用のカスタムエンドポイント
@app.post("/webrtc/offer")
async def custom_webrtc_offer(request_data: dict):
    """カスタムWebRTCオファーエンドポイント（エラーハンドリング付き）"""
    try:
        # FastRTCのデフォルト処理を呼び出し
        answer = await stream.offer(request_data)
        return answer
    except Exception as e:
        print(f"WebRTC offer error: {e}")
        # エラーが発生した場合でも適切なレスポンスを返す
        return {
            "type": "answer",
            "sdp": "",
            "error": str(e)
        }


# sample streaming endpoint ( https://www.ai-shift.co.jp/techblog/5680 )
@app.get("/outputs")
def _(webrtc_id: str):
    async def output_stream():
        try:
            async for output in stream.output_stream(webrtc_id):
                chatbot = output.args[0]
                yield f"event: output\ndata: {json.dumps(chatbot[-1])}\n\n"
        except Exception as e:
            print(f"Output stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")




def echoback(audio: tuple[int, np.ndarray]):
    print(f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]}秒の音声が入力されました。")
    # 音声認識
    user_input = stt_model.stt(audio)
    print(f"音声認識結果: {user_input}")
    
    # 認識した文章をそのまま音声合成してエコーバック
    response = user_input
    for audio_chunk in tts_model.stream_tts_sync(response, style_opt):
        print("Sending audio")
        yield audio_chunk

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

    stream.ui.launch(server_port=6076, server_name="0.0.0.0")

if __name__ == "__main__":
    example_echoback()