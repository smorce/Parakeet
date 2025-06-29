import nemo.collections.asr as nemo_asr
import sys, torch
import os
import librosa
import soundfile as sf
import tempfile
import gc
import math

# GPU/CPU自動判定とフォールバック
device = "cpu"
if torch.cuda.is_available():
    try:
        # CUDA初期化テスト
        torch.cuda.current_device()
        device = "cuda"
        print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name()}")
        # GPU メモリ情報を表示
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] Total GPU memory: {total_memory:.2f} GB")
    except RuntimeError as e:
        print(f"[WARNING] CUDA available but initialization failed: {e}")
        print("[INFO] Falling back to CPU")
        device = "cpu"
else:
    print("[INFO] CUDA not available, using CPU")

# CPUを強制する環境変数が設定されている場合
if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
    device = "cpu"
    print("[INFO] FORCE_CPU environment variable set, using CPU")

# デバイス設定
if device == "cpu":
    torch.set_default_device("cpu")

def load_model_with_fallback(device_preference="cuda"):
    """モデルを読み込み、メモリ不足時はCPUにフォールバック"""
    try:
        print(f"[INFO] Attempting to load model on {device_preference.upper()}")
        asr = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt_ctc-0.6b-ja"
        )
        
        if device_preference == "cuda" and torch.cuda.is_available():
            # GPU使用前にメモリをクリア
            torch.cuda.empty_cache()
            gc.collect()
            
            # モデルをGPUに移動
            asr = asr.cuda()
            
            # メモリ使用量をチェック
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"[INFO] GPU memory after model loading - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
            
            return asr, "cuda"
        else:
            asr = asr.cpu()
            return asr, "cpu"
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[WARNING] GPU out of memory during model loading: {e}")
            print("[INFO] Falling back to CPU")
            
            # GPU メモリをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # CPUでモデルを再読み込み
            asr = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt_ctc-0.6b-ja"
            )
            asr = asr.cpu()
            return asr, "cpu"
        else:
            raise e

def split_audio_for_memory_efficiency(audio_path, max_duration=30):
    """メモリ効率のため音声ファイルを分割"""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(y) / sr
    
    if duration <= max_duration:
        return [audio_path]
    
    # 分割数を計算
    num_splits = math.ceil(duration / max_duration)
    chunk_length = len(y) // num_splits
    
    temp_files = []
    for i in range(num_splits):
        start_idx = i * chunk_length
        end_idx = min((i + 1) * chunk_length, len(y))
        chunk = y[start_idx:end_idx]
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=f"_chunk_{i}.wav", delete=False) as tmpfile:
            sf.write(tmpfile.name, chunk, sr)
            temp_files.append(tmpfile.name)
    
    return temp_files

def transcribe_with_memory_management(asr, audio_path, device, max_duration=30):
    """メモリ効率的な音声転写（音声分割対応）"""
    try:
        # 長い音声ファイルは分割
        audio_chunks = split_audio_for_memory_efficiency(audio_path, max_duration)
        all_transcriptions = []
        
        for chunk_path in audio_chunks:
            try:
                # GPU使用時はメモリをクリア
                if device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 転写実行
                results = asr.transcribe([chunk_path])
                
                # 結果を取得してメモリをクリア
                for r in results:
                    all_transcriptions.append(r.text)
                
                # GPU使用時は再度メモリをクリア
                if device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device == "cuda":
                    print(f"[WARNING] GPU out of memory during chunk transcription: {e}")
                    print("[INFO] Falling back to CPU for this chunk")
                    
                    # GPU メモリをクリア
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # モデルを一時的にCPUに移動
                    asr_cpu = asr.cpu()
                    results = asr_cpu.transcribe([chunk_path])
                    for r in results:
                        all_transcriptions.append(r.text)
                    
                    # モデルを元のデバイスに戻す（次回のために）
                    if device == "cuda":
                        try:
                            asr.cuda()
                        except RuntimeError:
                            print("[WARNING] Cannot move model back to GPU, staying on CPU")
                            device = "cpu"
                else:
                    raise e
            
            finally:
                # 分割した一時ファイルをクリーンアップ
                if chunk_path != audio_path and os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        return all_transcriptions
        
    except Exception as e:
        # 分割ファイルのクリーンアップ
        for chunk_path in audio_chunks if 'audio_chunks' in locals() else []:
            if chunk_path != audio_path and os.path.exists(chunk_path):
                os.remove(chunk_path)
        raise e

try:
    asr, actual_device = load_model_with_fallback(device)
    print(f"[INFO] Model loaded successfully on {actual_device.upper()}")
    
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    sys.exit(1)

audio_files = sys.argv[1:]
if not audio_files:
    print("[ERROR] No audio files specified")
    sys.exit(1)

for audio_file in audio_files:
    print(f"[INFO] Processing: {audio_file}")
    try:
        # メモリ効率的な転写（音声分割対応）
        transcriptions = transcribe_with_memory_management(asr, audio_file, actual_device)
        
        for text in transcriptions:
            print(text)
            
    except Exception as e:
        print(f"[ERROR] Transcription failed for {audio_file}: {e}")
    finally:
        # メモリクリーンアップ
        gc.collect()
        if actual_device == "cuda":
            torch.cuda.empty_cache()