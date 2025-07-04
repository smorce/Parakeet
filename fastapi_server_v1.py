import tempfile
import os
import shutil
import subprocess
import librosa
import numpy as np
import soundfile as sf
import torch
import gc
import math
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import io
from pydantic import BaseModel, Field
from dataclasses import dataclass
from numpy.typing import NDArray
from fastapi.concurrency import run_in_threadpool

# --- FastRTCのインポート ---
from fastrtc import (
    Stream,
    ReplyOnPause,
    AlgoOptions,
    SileroVadOptions,
    AdditionalOutputs
)
from fastrtc.stream import Body

# --- Style-Bert-VITS2のインポート ---
from fastrtc_jp.text_to_speech.style_bert_vits2 import (
    StyleBertVits2,
    StyleBertVits2Options
)
from style_bert_vits2.constants import DEFAULT_STYLE
from style_bert_vits2.tts_model import TTSModel as SBV2_TTSModel

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPIアプリケーションの初期化 ---
app = FastAPI(
    title="NVIDIA Parakeet-TDT-CTC-0.6b-ja Transcription API",
    description="音声ファイルをアップロードして日本語の文字起こしを行うAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS設定 ---
# 異なるオリジン（ドメイン）で動作するフロントエンドからのリクエストを許可する
# 本番環境では、セキュリティのため allow_origins を特定のドメインに限定してください
# 例: allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可（開発用に便利）
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # OPTIONSを追加
    allow_headers=["*"], # 許可するHTTPヘッダー
)

# --- グローバル変数 ---
model = None
model_device = None
model_loaded = False
tts_model = None
tts_model_loaded = False

# --- デバイス設定 ---
def get_device():
    """利用可能なデバイス（CUDAまたはCPU）を判断して返す"""
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name()}")
            return "cuda"
        except RuntimeError as e:
            logger.warning(f"CUDA available but initialization failed: {e}")
            logger.warning("Falling back to CPU.")
            return "cpu"
    else:
        logger.info("CUDA not available, using CPU.")
        return "cpu"

DEVICE = get_device()

# --- モデル読み込み関数 ---
def load_model():
    """nvidia/parakeet-tdt_ctc-0.6b-jaモデルをメモリに読み込む"""
    global model, model_device, model_loaded
    if model_loaded:
        logger.info("Model is already loaded.")
        return

    logger.info(f"Attempting to load model on {DEVICE.upper()}...")
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt_ctc-0.6b-ja"
        )
        if DEVICE == "cuda":
            asr_model = asr_model.cuda()
        model = asr_model
        model_device = DEVICE
        model_loaded = True
        logger.info(f"Model loaded successfully on {model_device.upper()}.")
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
        # アプリケーション起動時にモデルが読み込めないと致命的なので、プロセスを終了させる
        raise RuntimeError(f"Model loading failed: {e}") from e

# --- TTS関連のカスタムクラスとPydanticモデル ---

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

class CustomStyleBertVits2(StyleBertVits2):
    """
    StyleBertVits2を継承し、推論時の引数をカスタマイズ可能にするクラス。
    """
    async def _run(self, text:str, options:CustomStyleBertVits2Options|None=None) -> tuple[int, NDArray[np.float32]]:
        model:SBV2_TTSModel = await self._load(options)
        speaker_id:int = 0 if 0 in model.id2spk else list(model.id2spk.keys())[0]
        speaker_style:str = DEFAULT_STYLE if DEFAULT_STYLE in model.style2id else list(model.style2id.keys())[0]

        if options:
            if options.speaker_id in model.id2spk:
                speaker_id = options.speaker_id
            elif options.speaker_name in model.spk2id:
                speaker_id = model.spk2id[options.speaker_name]
            if options.speaker_style in model.style2id:
                speaker_style = options.speaker_style

        infer_kwargs = {
            "speaker_id": speaker_id,
            "style": speaker_style,
            "assist_text": self._assist_text,
            "use_assist_text": True
        }

        if options:
            if options.sdp_ratio is not None: infer_kwargs["sdp_ratio"] = options.sdp_ratio
            if options.length_scale is not None: infer_kwargs["length"] = options.length_scale
            if options.noise_scale is not None: infer_kwargs["noise"] = options.noise_scale
            if options.pitch_scale is not None: infer_kwargs["pitch_scale"] = options.pitch_scale

        frame = model.infer(text, **infer_kwargs)
        self._assist_text = (self._assist_text + " " + text)[-200:]
        return frame

class TTSRequest(BaseModel):
    text: str = Field(..., description="音声合成するテキスト。", example="こんにちは、今日の天気はどうですか？")
    speaker_style: str | None = Field(None, description="話者のスタイルを指定します。", example="普通")
    sdp_ratio: float | None = Field(0.6, description="テンポの緩急を調整します (0.0 ~ 1.0)。", example=0.5)
    length_scale: float | None = Field(None, description="話速を調整します (1.0が基準)。", example=1.1)
    noise_scale: float | None = Field(None, description="音声のランダム性を調整します。", example=0.6)
    pitch_scale: float | None = Field(None, description="ピッチ（声の高さ）を調整します。", example=1.0)

# --- TTSモデル読み込み関数 ---
def load_tts_model():
    """Style-Bert-VITS2モデルをメモリに読み込む"""
    global tts_model, tts_model_loaded
    if tts_model_loaded:
        logger.info("TTS Model is already loaded.")
        return

    logger.info(f"Attempting to load TTS model on {DEVICE.upper()}...")
    try:
        # TODO: 以下のパスは環境に合わせて修正してください
        tts_model_path = "models/style_bert_vits2/model.safetensors"
        tts_config_path = "models/style_bert_vits2/config.json"
        tts_style_vec_path = "models/style_bert_vits2/style_vectors.npy"
        
        if not os.path.exists(tts_model_path) or not os.path.exists(tts_config_path):
             logger.warning("TTS model files not found. Skipping TTS model loading.")
             logger.warning(f"Please place model files at '{tts_model_path}' and '{tts_config_path}'")
             tts_model_loaded = False
             return

        tts_model = CustomStyleBertVits2()
        # オプションを設定してモデルをロード
        # ttsメソッド呼び出し時に毎回ロードするのではなく、ここで一度だけロードする
        # ここでのオプションはデフォルトとして機能する
        style_opt = CustomStyleBertVits2Options(
            device=DEVICE,
            model_path=tts_model_path,
            config_path=tts_config_path,
            style_vec_path=tts_style_vec_path,
        )
        # _loadを呼び出して実際にモデルをメモリに読み込ませる
        # StyleBertVits2の内部実装では、最初の呼び出し時にロードが行われるため、
        # ここでダミーのテキストで一度呼び出しておく
        # これは非同期メソッドなので、イベントループで実行する必要がある
        asyncio.run(tts_model.tts("モデル読み込み中", options=style_opt))

        tts_model_loaded = True
        logger.info(f"TTS Model loaded successfully on {DEVICE.upper()}.")

    except Exception as e:
        logger.error(f"Failed to load the TTS model: {e}", exc_info=True)
        # TTSモデルの読み込みは必須ではないため、エラーを発生させずに警告に留める
        tts_model_loaded = False
        logger.warning("TTS functionality will be unavailable.")

# --- 音声処理・文字起こし関数 (変更なし) ---
def split_audio(audio_path: str, max_duration: int = 30):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(y) / sr
    if duration <= max_duration:
        return [audio_path]
    num_splits = math.ceil(duration / max_duration)
    chunk_length = len(y) // num_splits
    temp_files = []
    logger.info(f"Splitting audio into {num_splits} chunks.")
    for i in range(num_splits):
        start_idx = i * chunk_length
        end_idx = min((i + 1) * chunk_length, len(y))
        chunk = y[start_idx:end_idx]
        with tempfile.NamedTemporaryFile(suffix=f"_chunk_{i}.wav", delete=False) as tmpfile:
            sf.write(tmpfile.name, chunk, sr)
            temp_files.append(tmpfile.name)
    return temp_files

def transcribe_audio_chunks(asr_model, audio_chunks: list[str]):
    all_transcriptions = []
    try:
        for i, chunk_path in enumerate(audio_chunks):
            logger.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")
            if model_device == "cuda": torch.cuda.empty_cache()
            results = asr_model.transcribe([chunk_path])
            if results and isinstance(results, list) and hasattr(results[0], 'text'):
                all_transcriptions.append(results[0].text)
        return all_transcriptions
    finally:
        for path in audio_chunks:
            if "_chunk_" in path and os.path.exists(path): os.remove(path)
        if model_device == "cuda": torch.cuda.empty_cache()

# --- リアルタイム音声処理ハンドラ ---
def handle_audio_realtime(audio: tuple[int, np.ndarray]):
    """
    FastRTCから受け取った音声チャンクを処理し、文字起こしを実行する。
    """
    global model
    sr, audio_data = audio

    if not model_loaded or model is None:
        logger.error("Model is not loaded, cannot transcribe.")
        # モデルがなければ処理を中断
        return

    # 音声データがfloat32でない場合は変換
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    # 一時ファイルに音声データを書き込む
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, sr)
        
        logger.info(f"Transcribing audio chunk from {tmp_path}...")
        if model_device == "cuda": torch.cuda.empty_cache()

        # nemo-asrのtranscribeメソッドは同期的であるため、
        # asyncio.to_thread を使って別スレッドで実行することも検討できるが、
        # ここではシンプルに直接呼び出す。GPU処理が高速なため影響は限定的と想定。
        transcriptions = model.transcribe([tmp_path])
        
        if transcriptions and isinstance(transcriptions, list) and len(transcriptions) > 0:
            result_text = transcriptions[0]
            logger.info(f"Transcription result: {result_text}")
            # 文字起こし結果をAdditionalOutputsでクライアントに送る
            yield AdditionalOutputs(result_text)

    except Exception as e:
        logger.error(f"Error during real-time transcription: {e}", exc_info=True)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if model_device == "cuda": torch.cuda.empty_cache()
        gc.collect()

def handle_audio_clone(audio: tuple[int, np.ndarray]):
    """
    ボイスクローニング用の音声処理ハンドラ。
    STT -> TTS の順に処理し、合成音声をストリーミングで返す。
    """
    global model, tts_model
    sr, audio_data = audio

    # --- ボイスクローニング（Talk-back）用の設定 ---
    clone_style_options = CustomStyleBertVits2Options(
        device=DEVICE,
        model_path="models/style_bert_vits2/model.safetensors",
        config_path="models/style_bert_vits2/config.json",
        style_vec_path="models/style_bert_vits2/style_vectors.npy",
        speaker_style="上機嫌", # デフォルトのスタイル
        sdp_ratio=0.6
    )

    if not model_loaded or not tts_model_loaded:
        logger.warning("ASR or TTS model is not loaded. Skipping clone process.")
        return

    # 1. 音声認識 (STT)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    
    user_text = ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, sr)
        
        transcriptions = model.transcribe([tmp_path])
        if transcriptions and isinstance(transcriptions, list) and len(transcriptions) > 0:
            user_text = transcriptions[0]
            logger.info(f"[Clone] STT Result: {user_text}")
            # 文字起こし結果をSSEでクライアントに通知
            yield AdditionalOutputs(user_text)
        else:
            return # 文字起こし結果がなければ何もしない
    except Exception as e:
        logger.error(f"[Clone] Error during STT: {e}", exc_info=True)
        return
    finally:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

    # 2. 音声合成 (TTS)
    if user_text:
        try:
            logger.info(f"[Clone] Starting TTS for text: {user_text}")
            # stream_tts_syncは合成音声のチャンクを返すジェネレータ
            for chunk in tts_model.stream_tts_sync(user_text, options=clone_style_options):
                yield chunk  # 音声チャンクをクライアントにストリーミング
            logger.info("[Clone] TTS streaming finished.")
        except Exception as e:
            logger.error(f"[Clone] Error during TTS streaming: {e}", exc_info=True)

# --- FastRTC Streamの作成 ---

# 1. 既存のリアルタイム文字起こし用Stream
stream = Stream(
    handler=ReplyOnPause(
        handle_audio_realtime,
        can_interrupt=True,
        algo_options=AlgoOptions(
            # パラメータはユースケースに合わせて調整
            audio_chunk_duration=0.8,
            started_talking_threshold=0.3,
            speech_threshold=0.2
        ),
        model_options=SileroVadOptions(
            # VADの感度設定
            threshold=0.4,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500 # 無音区間を少し長めに設定して、文の区切りを捉えやすくする
        )
    ),
    modality="audio",
    mode="send-receive" # クライアントから音声を受信し、サーバーからデータ（空の音声）を返す
)

# 2. 新しいボイスクローニング用Stream
stream_clone = Stream(
    handler=ReplyOnPause(
        handle_audio_clone,
        can_interrupt=True,
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,
            started_talking_threshold=0.2,
            speech_threshold=0.1
        ),
        model_options=SileroVadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )
    ),
    modality="audio",
    mode="send-receive"
)

# --- FastAPIエンドポイント ---
@app.on_event("startup")
def startup_event():
    """FastAPI起動時にモデルを読み込む"""
    load_model()
    load_tts_model()

# 古い/transcribeエンドポイントは削除

# --- SSEエンドポイント：文字起こし結果をストリーミング配信 ---
@app.get("/transcribe-stream")
async def transcribe_stream_endpoint(webrtc_id: str):
    """
    指定されたwebrtc_idの文字起こし結果をServer-Sent Eventsでストリーミング配信する
    """
    async def event_generator():
        logger.info(f"SSE stream started for webrtc_id: {webrtc_id}")
        try:
            async for output in stream.output_stream(webrtc_id):
                if isinstance(output, AdditionalOutputs) and output.args:
                    transcript_text = output.args[0]
                    logger.info(f"Sending data to client {webrtc_id}: {transcript_text}")
                    yield f"data: {transcript_text}\n\n"
        except asyncio.CancelledError:
             logger.info(f"Client {webrtc_id} disconnected from SSE stream.")
        except Exception as e:
            logger.error(f"Error in SSE stream for {webrtc_id}: {e}", exc_info=True)
        finally:
            logger.info(f"SSE stream closed for webrtc_id: {webrtc_id}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/clone-transcribe-stream")
async def clone_transcribe_stream_endpoint(webrtc_id: str):
    """
    ボイスクローニングセッションの文字起こし結果をSSEでストリーミング配信する
    """
    async def event_generator():
        logger.info(f"Clone SSE stream started for webrtc_id: {webrtc_id}")
        try:
            async for output in stream_clone.output_stream(webrtc_id):
                if isinstance(output, AdditionalOutputs) and output.args:
                    transcript_text = output.args[0]
                    logger.info(f"Sending clone data to client {webrtc_id}: {transcript_text}")
                    yield f"data: {transcript_text}\n\n"
        except asyncio.CancelledError:
             logger.info(f"Client {webrtc_id} disconnected from Clone SSE stream.")
        except Exception as e:
            logger.error(f"Error in Clone SSE stream for {webrtc_id}: {e}", exc_info=True)
        finally:
            logger.info(f"Clone SSE stream closed for webrtc_id: {webrtc_id}")
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/synthesize/", response_class=StreamingResponse)
async def synthesize_endpoint(request: TTSRequest):
    """
    テキストを受け取り、Style-Bert-VITS2で音声を合成してWAVファイルとして返します。
    """
    if not tts_model_loaded or tts_model is None:
        raise HTTPException(status_code=503, detail="TTS Model is not loaded or unavailable.")

    try:
        # TODO: 将来的にモデルパスなどをリクエストで指定できるようにする
        options = CustomStyleBertVits2Options(
            device=DEVICE,
            model_path="models/style_bert_vits2/model.safetensors", # 読み込み時と同じパスを指定
            config_path="models/style_bert_vits2/config.json",   # 読み込み時と同じパスを指定
            style_vec_path="models/style_bert_vits2/style_vectors.npy", # 読み込み時と同じパスを指定
            speaker_style=request.speaker_style,
            sdp_ratio=request.sdp_ratio,
            length_scale=request.length_scale,
            noise_scale=request.noise_scale,
            pitch_scale=request.pitch_scale
        )
        
        logger.info(f"Synthesizing speech for text: '{request.text[:30]}...'")

        # run_in_threadpoolを使用して、ブロッキングI/Oである音声合成を非同期で実行
        sample_rate, audio_data = await run_in_threadpool(
            tts_model.tts, request.text, options=options
        )
        
        # メモリ上でWAVファイルを作成
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
        wav_buffer.seek(0)
        
        logger.info("Speech synthesis successful.")

        return StreamingResponse(wav_buffer, media_type="audio/wav")

    except Exception as e:
        logger.error(f"An error occurred during speech synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to synthesize audio: {str(e)}")

@app.get("/health", status_code=200)
def health_check():
    """APIのヘルスチェック用エンドポイント"""
    return {"status": "ok", "model_loaded": model_loaded, "device": model_device, "tts_model_loaded": tts_model_loaded}

# --- FastRTCのエンドポイントを自前で定義 ---
@app.post("/webrtc/offer")
async def webrtc_offer(body: Body):
    """
    クライアントからWebRTCのオファーを受け取り、アンサーを返す
    """
    try:
        answer = await stream.offer(body)
        return answer
    except Exception as e:
        logger.error(f"Error processing WebRTC offer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clone/webrtc/offer")
async def clone_webrtc_offer(body: Body):
    """
    クライアントからボイスクローニング用のWebRTCオファーを受け取り、アンサーを返す
    """
    try:
        answer = await stream_clone.offer(body)
        return answer
    except Exception as e:
        logger.error(f"Error processing clone WebRTC offer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Uvicornで実行 ---
if __name__ == "__main__":
    import uvicorn
    # reload=Trueは開発時には便利だが、本番環境ではFalseを推奨
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=5466, reload=True)