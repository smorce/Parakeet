import nemo.collections.asr as nemo_asr
import sys, torch
import os
import librosa
import soundfile as sf
import tempfile
import gc
import math
import gradio as gr

# グローバル変数でモデルとデバイス状態を管理
model = None
model_device = None
model_loaded = False

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

def load_model_with_fallback(device_preference="cuda", progress=gr.Progress()):
    """モデルを読み込み、メモリ不足時はCPUにフォールバック"""
    global model, model_device, model_loaded
    
    if model_loaded and model is not None:
        return model, model_device
    
    try:
        progress(0.1, desc=f"{device_preference.upper()}でモデルを読み込み中...")
        print(f"[INFO] Attempting to load model on {device_preference.upper()}")
        
        progress(0.3, desc="Parakeet-TDT-CTCモデルをダウンロード中...")
        asr = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt_ctc-0.6b-ja"
        )
        
        if device_preference == "cuda" and torch.cuda.is_available():
            progress(0.6, desc="モデルをGPUに移動中...")
            # GPU使用前にメモリをクリア
            torch.cuda.empty_cache()
            gc.collect()
            
            # モデルをGPUに移動
            asr = asr.cuda()
            
            # メモリ使用量をチェック
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"[INFO] GPU memory after model loading - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
            
            model = asr
            model_device = "cuda"
            model_loaded = True
            progress(1.0, desc="モデル読み込み完了（GPU）")
            return asr, "cuda"
        else:
            progress(0.8, desc="モデルをCPUに設定中...")
            asr = asr.cpu()
            model = asr
            model_device = "cpu"
            model_loaded = True
            progress(1.0, desc="モデル読み込み完了（CPU）")
            return asr, "cpu"
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[WARNING] GPU out of memory during model loading: {e}")
            print("[INFO] Falling back to CPU")
            progress(0.7, desc="GPU メモリ不足のためCPUにフォールバック中...")
            
            # GPU メモリをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # CPUでモデルを再読み込み
            asr = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt_ctc-0.6b-ja"
            )
            asr = asr.cpu()
            model = asr
            model_device = "cpu"
            model_loaded = True
            progress(1.0, desc="モデル読み込み完了（CPU）")
            return asr, "cpu"
        else:
            progress(1.0, desc=f"モデル読み込みエラー: {e}")
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

def transcribe_with_memory_management(asr, audio_path, device, max_duration=30, progress=gr.Progress()):
    """メモリ効率的な音声転写（音声分割対応）"""
    try:
        progress(0.1, desc="音声ファイルを分析中...")
        
        # 長い音声ファイルは分割
        audio_chunks = split_audio_for_memory_efficiency(audio_path, max_duration)
        all_transcriptions = []
        
        total_chunks = len(audio_chunks)
        print(f"[INFO] Processing {total_chunks} audio chunks")
        
        for i, chunk_path in enumerate(audio_chunks):
            try:
                # 進捗更新
                chunk_progress = 0.2 + (0.7 * i / total_chunks)
                progress(chunk_progress, desc=f"音声を文字起こし中... ({i+1}/{total_chunks})")
                
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
                    
                    progress(chunk_progress, desc=f"GPU メモリ不足のためCPUにフォールバック中... ({i+1}/{total_chunks})")
                    
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
        
        progress(0.95, desc="結果をまとめ中...")
        return all_transcriptions
        
    except Exception as e:
        # 分割ファイルのクリーンアップ
        for chunk_path in audio_chunks if 'audio_chunks' in locals() else []:
            if chunk_path != audio_path and os.path.exists(chunk_path):
                os.remove(chunk_path)
        raise e

def process_audio_file(audio_file, progress=gr.Progress()):
    """音声ファイルを処理して文字起こしを実行"""
    if audio_file is None:
        return "音声ファイルをアップロードしてください。"
    
    try:
        # モデルが読み込まれていない場合は読み込む
        global model, model_device
        if not model_loaded:
            progress(0.0, desc="モデルを読み込み中...")
            model, model_device = load_model_with_fallback(device, progress)
        
        # 音声ファイルのパスを取得
        audio_path = audio_file
        if hasattr(audio_file, 'name'):
            audio_path = audio_file.name
        
        print(f"[INFO] Processing: {audio_path}")
        
        # メモリ効率的な転写（音声分割対応）
        transcriptions = transcribe_with_memory_management(model, audio_path, model_device, progress=progress)
        
        # 結果を結合
        final_result = "\n".join(transcriptions)
        
        progress(1.0, desc="文字起こし完了！")
        
        return final_result
            
    except Exception as e:
        error_msg = f"文字起こしに失敗しました: {e}"
        print(f"[ERROR] {error_msg}")
        return error_msg
    finally:
        # メモリクリーンアップ
        gc.collect()
        if model_device == "cuda":
            torch.cuda.empty_cache()

# Gradio インターフェースの作成
def create_gradio_interface():
    """Gradio インターフェースを作成"""
    
    with gr.Blocks(title="音声文字起こしシステム", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎤 音声文字起こしシステム
        
        NVIDIA Parakeet-TDT-CTCモデルを使用した日本語音声文字起こしツールです。
        音声ファイルをアップロードして「文字起こし実行」ボタンを押してください。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 音声ファイルアップロード
                audio_input = gr.Audio(
                    label="音声ファイル",
                    type="filepath",
                    sources=["upload"]
                )
                
                # 実行ボタン
                transcribe_btn = gr.Button(
                    "🚀 文字起こし実行",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # 結果表示
                output_text = gr.Textbox(
                    label="文字起こし結果",
                    lines=15,
                    max_lines=30,
                    placeholder="ここに文字起こし結果が表示されます...",
                    show_copy_button=True
                )
        
        # システム情報表示
        gr.Markdown(f"""
        ### システム情報
        - **使用デバイス**: {device.upper()}
        - **モデル**: nvidia/parakeet-tdt_ctc-0.6b-ja, silero-vad
        - **対応フォーマット**: WAV, MP3, FLAC, M4A など
        """)
        
        # イベントハンドラ
        transcribe_btn.click(
            fn=process_audio_file,
            inputs=[audio_input],
            outputs=[output_text],
            show_progress=True
        )
    
    return interface

if __name__ == "__main__":
    # コマンドライン引数がある場合は従来の処理を実行
    if len(sys.argv) > 1:
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
    else:
        # Gradio UIを起動
        print("[INFO] Starting Gradio interface...")
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=3791,
            share=False,
            show_error=True
        )