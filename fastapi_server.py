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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

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
    allow_methods=["GET", "POST"], # 許可するHTTPメソッド
    allow_headers=["*"], # 許可するHTTPヘッダー
)

# --- グローバル変数 ---
model = None
model_device = None
model_loaded = False

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
        raise RuntimeError(f"Model loading failed: {e}") from e

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

# --- FastAPIエンドポイント ---
@app.on_event("startup")
def startup_event():
    """FastAPI起動時にモデルを読み込む"""
    load_model()

@app.post("/transcribe/", response_class=JSONResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    max_duration: int = Form(30)
):
    """音声ファイルをアップロードして文字起こしを実行する"""
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please try again later.")

    # 一時ファイルとしてWebMを保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
        shutil.copyfileobj(file.file, tmp_webm)
        webm_path = tmp_webm.name

    wav_path = None # finallyブロックで使うために初期化
    try:
        # ffmpegを使ってWebMをWAVに変換
        wav_path = webm_path.replace(".webm", ".wav")
        command = [
            "ffmpeg",
            "-i", webm_path,      # 入力ファイル
            "-vn",                # ビデオストリームを無視
            "-acodec", "pcm_s16le",# WAVのコーデック
            "-ar", "16000",       # サンプリングレートを16kHzに
            "-ac", "1",           # モノラルに
            wav_path
        ]
        
        logger.info(f"Converting {webm_path} to {wav_path} using ffmpeg...")
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        # 音声が処理可能かチェック (変換後のWAVファイルに対して)
        try:
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            logger.warning(f"Could not load converted audio file {wav_path}, skipping. Error: {e}")
            return JSONResponse(status_code=200, content={"transcription": ""})

        # 無音に近い場合はスキップ
        if duration < 0.1:
            logger.info(f"Audio chunk duration ({duration:.2f}s) is too short. Skipping.")
            return JSONResponse(status_code=200, content={"transcription": ""})
        
        # 既存の処理をWAVファイルに対して実行
        # split_audioは既にパスを受け取るので、そのまま使える
        audio_chunks = split_audio(wav_path, max_duration)
        transcriptions = transcribe_audio_chunks(model, audio_chunks)
        final_result = "\n".join(transcriptions)
        
        return JSONResponse(status_code=200, content={"transcription": final_result})

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed. Error: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"ffmpeg conversion failed: {e.stderr}")
    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 一時ファイルをクリーンアップ
        if os.path.exists(webm_path):
            os.remove(webm_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        gc.collect()

@app.get("/health", status_code=200)
def health_check():
    """APIのヘルスチェック用エンドポイント"""
    return {"status": "ok", "model_loaded": model_loaded, "device": model_device}

# --- Uvicornで実行 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=5466, reload=True)