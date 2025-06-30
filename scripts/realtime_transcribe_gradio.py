import gradio as gr
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from queue import Queue, Empty
from threading import Thread, Event
import time
import os
import torchaudio

# --- グローバル変数と定数 ---
SAMPLING_RATE = 16000
AUDIO_QUEUE = Queue()
RESULT_QUEUE = Queue()
STOP_EVENT = Event()

# --- モデルのロード ---
def load_model_with_retry(model_loader, model_name, max_retries=3):
    """リトライ機能付きのモデル読み込み"""
    for attempt in range(max_retries):
        try:
            print(f"[INFO] モデル読み込み試行 {attempt + 1}/{max_retries}: {model_name}")
            return model_loader()
        except Exception as e:
            error_msg = str(e).lower()
            print(f"[WARNING] 試行 {attempt + 1} 失敗: {e}")
            
            if 'rate limit' in error_msg or '403' in error_msg or '502' in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30
                    print(f"[INFO] 一時的なエラーのため、{wait_time}秒待機してから再試行します...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[ERROR] 最大リトライ回数に達しました。エラーが継続しています。")
                    raise e
            else: # その他のエラー
                raise e
    raise Exception(f"モデル {model_name} の読み込みに複数回失敗しました。")


def load_models():
    """ASRモデルとVADモデルをロードする"""
    print("[INFO] モデルをロードしています...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用デバイス: {device}")
    
    try:
        # ASRモデルのローダー
        def asr_loader():
            return nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-0.6b-ja").to(device)
        
        # VADモデルのローダー
        def vad_loader():
            return torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)

        print("[INFO] NeMo ASRモデル (nvidia/parakeet-tdt_ctc-0.6b-ja) をロード中...")
        asr_model = load_model_with_retry(asr_loader, "NeMo ASR")
        print("[INFO] ASRモデルのロード完了。")
        
        print("[INFO] Silero VADモデルをロード中...")
        vad_model, utils = load_model_with_retry(vad_loader, "Silero VAD")
        print("[INFO] VADモデルのロード完了。")

        return asr_model, vad_model, utils, device

    except Exception as e:
        print(f"[FATAL] モデルのロードに失敗しました: {e}")
        return None, None, None, None

asr_model, vad_model, vad_utils, device = load_models()
(VADIterator,) = (vad_utils[3],) if vad_utils else (None,)

# --- 音声処理スレッド ---
def processing_thread():
    """音声キューからデータを取り出し、VADと文字起こしを実行するスレッド"""
    print("[INFO] 処理スレッドを開始しました。")
    # VADイテレータを初期化（感度調整のためthresholdを設定）
    vad_iterator = VADIterator(vad_model, threshold=0.7) if VADIterator else None
    speech_buffer = []
    
    while not STOP_EVENT.is_set():
        try:
            # キューから単一の音声チャンクを取得
            audio_item = AUDIO_QUEUE.get(timeout=0.2)

            # (sampling_rate, audio_np) 形式を想定
            if isinstance(audio_item, tuple):
                orig_sr, audio_chunk_np = audio_item
            else:
                orig_sr, audio_chunk_np = SAMPLING_RATE, audio_item

            # 必要に応じてリサンプリング (torchaudioは torch.Tensor を想定)
            if orig_sr != SAMPLING_RATE:
                audio_tensor_rs = torchaudio.functional.resample(
                    torch.from_numpy(audio_chunk_np).float(),
                    orig_sr,
                    SAMPLING_RATE,
                )
                audio_chunk_np = audio_tensor_rs.numpy()

            # Silero VAD は 16kHz で 512 サンプル(約32ms) を想定
            FRAME_SIZE = 512  # 16kHz -> 32ms

            offset = 0
            total_len = len(audio_chunk_np)

            while offset + FRAME_SIZE <= total_len:
                frame_np = audio_chunk_np[offset : offset + FRAME_SIZE]
                offset += FRAME_SIZE

                if vad_iterator:
                    speech_buffer.append(frame_np)
                    audio_frame_tensor = torch.from_numpy(frame_np).float()
                    speech_dict = vad_iterator(audio_frame_tensor, return_seconds=True)

                    if speech_dict and 'end' in speech_dict:
                        full_audio_np = np.concatenate(speech_buffer)
                        if len(full_audio_np) > SAMPLING_RATE * 0.4:
                            transcribe_audio(full_audio_np)
                        speech_buffer.clear()
                        vad_iterator.reset_states()

                else:
                    transcribe_audio(frame_np)

        except Empty:
            # キューが空の場合はループを継続
            continue
        except Exception as e:
            print(f"[ERROR] 処理スレッドでエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
            
    print("[INFO] 処理スレッドを停止しました。")

def transcribe_audio(audio_data):
    """音声データを文字起こしする共通関数"""
    try:
        audio_tensor = torch.from_numpy(audio_data).float().to(device).unsqueeze(0)
        with torch.no_grad():
            transcriptions = asr_model.transcribe(audio_tensor, verbose=False)
        
        if transcriptions and transcriptions[0]:
            RESULT_QUEUE.put(transcriptions[0])
            print(f"[INFO] 文字起こし結果: {transcriptions[0]}")
    except Exception as e:
        print(f"[ERROR] 文字起こし中にエラーが発生しました: {e}")

# --- Gradioインターフェース ---
processing_thread_instance = None

def start_transcription():
    """文字起こし開始"""
    global processing_thread_instance
    if processing_thread_instance is None or not processing_thread_instance.is_alive():
        STOP_EVENT.clear()
        # キューと状態をクリア
        while not AUDIO_QUEUE.empty(): AUDIO_QUEUE.get_nowait()
        while not RESULT_QUEUE.empty(): RESULT_QUEUE.get_nowait()
        
        processing_thread_instance = Thread(target=processing_thread)
        processing_thread_instance.start()
        print("[INFO] 文字起こしプロセスを開始しました。")
        # status_box, text_state, transcript_box の値を返す
        return "状態: 実行中", "", ""
    return "状態: 既に実行中です", "", ""

def stop_transcription(current_text):
    """文字起こし停止"""
    global processing_thread_instance
    if processing_thread_instance and processing_thread_instance.is_alive():
        STOP_EVENT.set()
        processing_thread_instance.join(timeout=2)
        processing_thread_instance = None
        print("[INFO] 文字起こしプロセスを停止しました。")

        # 停止後に残っているキューのアイテムを処理
        final_text = current_text
        while not RESULT_QUEUE.empty():
            final_text += RESULT_QUEUE.get_nowait() + " "
        return "状態: 停止", final_text
    return "状態: 既に停止しています", current_text

def audio_stream_callback(current_text, audio_chunk):
    """ブラウザから送られてくる音声チャンクをキューに積み、文字起こし結果でUIを更新"""
    if audio_chunk is not None:
        # audio_chunkがタプル形式 (sampling_rate, audio_data) の場合の処理
        if isinstance(audio_chunk, tuple):
            sampling_rate, audio_data = audio_chunk
            print(f"[DEBUG] Audio chunk received - shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'no shape'}, type: {type(audio_data)}")
            if audio_data.dtype == np.int16:
                y = audio_data.astype(np.float32) / 32767.0
            else:
                y = audio_data.astype(np.float32)

            # 大きすぎるチャンク（録音完了後に送られてくる全バッファなど）は無視する
            if len(y) > SAMPLING_RATE * 3:
                return current_text, current_text
        else:
            # audio_chunkがNumPy配列の場合の処理
            print(f"[DEBUG] Audio chunk received - shape: {audio_chunk.shape if hasattr(audio_chunk, 'shape') else 'no shape'}, type: {type(audio_chunk)}")
            if audio_chunk.dtype == np.int16:
                y = audio_chunk.astype(np.float32) / 32767.0
            else:
                y = audio_chunk.astype(np.float32)
            sampling_rate = SAMPLING_RATE

            if len(y) > SAMPLING_RATE * 3:
                return current_text, current_text

        # (sampling_rate, audio_data) をキューに追加
        AUDIO_QUEUE.put((sampling_rate, y))

    # 結果キューに溜まっている文字列をすべて取り出して返す
    new_text = ""
    while not RESULT_QUEUE.empty():
        new_text += RESULT_QUEUE.get_nowait() + " "
    
    updated_text = current_text + new_text
    return updated_text, updated_text

with gr.Blocks() as demo:
    gr.Markdown("# リアルタイム文字起こし (ブラウザマイク版)")
    gr.Markdown("マイクの使用を許可し、「録音開始」ボタンを押してください。")
    
    with gr.Row():
        status_box = gr.Textbox("状態: 停止", label="ステータス", interactive=False)
    
    # 文字起こし結果を保持するための状態
    text_state = gr.State("")
    
    audio_input = gr.Audio(sources=["microphone"], type="numpy", label="マイク入力")
    transcript_box = gr.Textbox(label="文字起こし結果", lines=10, interactive=False)
    
    start_button = gr.Button("録音開始", variant="primary")
    stop_button = gr.Button("録音停止")

    # イベントハンドラ
    stream_event = audio_input.stream(
        fn=audio_stream_callback,
        inputs=[text_state, audio_input],
        outputs=[text_state, transcript_box],
    )

    start_button.click(
        fn=start_transcription,
        outputs=[status_box, text_state, transcript_box]
    )

    stop_button.click(
        fn=stop_transcription,
        inputs=[text_state],
        outputs=[status_box, transcript_box],
        cancels=[stream_event]
    )

if __name__ == "__main__":
    if asr_model is None:
        print("[FATAL] モデルのロードに失敗したため、アプリケーションを起動できません。")
    else:
        print("[INFO] Gradioアプリケーションを起動します...")
        demo.launch(server_name="0.0.0.0", server_port=3791) 