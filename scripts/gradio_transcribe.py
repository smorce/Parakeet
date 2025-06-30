import nemo.collections.asr as nemo_asr
import sys, torch
import os
import librosa
import soundfile as sf
import tempfile
import gc
import math
import gradio as gr
import speech_recognition as sr
# 注意: リアルタイム機能はブラウザネイティブのJavaScriptで実装されているため、
# Python側での音声処理ライブラリは不要になりました

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

# 新しいマイク音声ファイルを文字起こしする関数
def transcribe_mic_audio(audio_file, progress=gr.Progress()):
    """ブラウザで録音された音声ファイル（WAV/OGGなど）を文字起こし"""
    if audio_file is None:
        return "録音された音声がありません。"

    try:
        recognizer = sr.Recognizer()
        progress(0.2, desc="音声を読み込み中...")
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        progress(0.6, desc="音声を認識中...")
        text = recognizer.recognize_google(audio_data, language='ja-JP')
        progress(1.0, desc="認識完了")
        return text
    except sr.UnknownValueError:
        return "音声を認識できませんでした。"
    except sr.RequestError as e:
        return f"Google Web Speech APIに接続できませんでした: {e}"
    except Exception as e:
        return f"エラーが発生しました: {e}"

# 注意: 以前のPythonベースのリアルタイム機能は、
# ブラウザネイティブのJavaScript実装に置き換えられました

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
        model_was_loaded = model_loaded
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
def get_system_info():
    """システム情報を取得"""
    info = {
        "device": device.upper(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpu_name": "None",
        "total_memory": "N/A",
        "allocated_memory": "N/A",
        "cached_memory": "N/A"
    }
    
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["total_memory"] = f"{total_memory:.2f} GB"
            
            if model_loaded:
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                info["allocated_memory"] = f"{allocated:.2f} GB"
                info["cached_memory"] = f"{cached:.2f} GB"
            else:
                info["allocated_memory"] = "モデル未読み込み"
                info["cached_memory"] = "モデル未読み込み"
        except:
            pass
    
    return info

def create_gradio_interface():
    """Gradio インターフェースを作成"""
    
    # ブラウザネイティブリアルタイム音声認識用のJavaScript
    realtime_js = """
    <script>
    // リアルタイム音声認識の状態管理
    let realtimeState = {
        recognition: null,
        stream: null,
        isActive: false,
        finalTranscript: '',
        results: []
    };

    // Web Speech API サポートチェック
    function checkWebSpeechSupport() {
        return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
    }

    // ステータス更新関数
    function updateStatus(message) {
        const statusElement = document.getElementById('realtime_status');
        if (statusElement && statusElement.querySelector('textarea')) {
            statusElement.querySelector('textarea').value = message;
        }
    }

    // 結果表示更新関数
    function updateResults(transcript, isFinal) {
        const outputElement = document.getElementById('realtime_output');
        if (outputElement && outputElement.querySelector('textarea')) {
            const textarea = outputElement.querySelector('textarea');
            const timestamp = new Date().toLocaleTimeString('ja-JP');
            
            if (isFinal) {
                realtimeState.finalTranscript += transcript + '\\n';
                textarea.value = realtimeState.finalTranscript;
            } else {
                textarea.value = realtimeState.finalTranscript + transcript;
            }
            
            // 自動スクロール
            textarea.scrollTop = textarea.scrollHeight;
            
            // 最終結果のみキャッシュ
            if (isFinal) {
                realtimeState.results.push(`[${timestamp}] ${transcript}`);
            }
        }
    }
    
    // Web Speech Recognition設定
    function setupSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            updateStatus('❌ Web Speech APIがサポートされていません');
            return false;
        }
        
        realtimeState.recognition = new SpeechRecognition();
        realtimeState.recognition.lang = 'ja-JP';
        realtimeState.recognition.interimResults = true;
        realtimeState.recognition.continuous = true;
        
        realtimeState.recognition.onstart = function() {
            updateStatus('🎤 音声認識中... 話しかけてください');
        };
        
        realtimeState.recognition.onresult = function(event) {
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    updateResults(event.results[i][0].transcript, true);
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            if (interimTranscript) {
                updateResults(interimTranscript, false);
            }
        };
        
        realtimeState.recognition.onerror = function(event) {
            // 'aborted' は stopRealtimeRecognition() による正常終了なので無視する
            if (event.error === 'aborted') {
                updateStatus('⏹️ リアルタイム認識を停止しました');
                return;
            }
            console.error('音声認識エラー:', event.error);
            updateStatus(`❌ エラー: ${event.error}`);
        };
        
        realtimeState.recognition.onend = function() {
            if (realtimeState.isActive) {
                realtimeState.recognition.start(); // 連続認識
            } else {
                updateStatus('⏹️ リアルタイム認識を停止しました');
            }
        };
        
        return true;
    }

    // グローバルスコープに関数を配置
    window.startRealtimeRecognition = async function() {
        try {
            if (realtimeState.isActive) {
                console.warn("Recognition already active");
                return;
            }
            if (!checkWebSpeechSupport()) {
                updateStatus('❌ Web Speech APIがサポートされていません');
                return;
            }
            
            updateStatus('🎤 マイクへのアクセスを要求中...');
            
            realtimeState.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            if (!setupSpeechRecognition()) {
                return;
            }
            
            realtimeState.isActive = true;
            realtimeState.recognition.start();
            
        } catch (error) {
            console.error('リアルタイム認識開始エラー:', error);
            updateStatus(`❌ エラー: ${error.message}`);
        }
    }

    window.stopRealtimeRecognition = function() {
        realtimeState.isActive = false;
        if (realtimeState.recognition) {
            // abort() は 'aborted' エラーイベントを発火するため、stop() を使用して穏やかに終了
            realtimeState.recognition.stop();
        }
        if (realtimeState.stream) {
            realtimeState.stream.getTracks().forEach(track => track.stop());
        }
    }

    window.clearResults = function() {
        const outputElement = document.getElementById('realtime_output');
        if (outputElement && outputElement.querySelector('textarea')) {
            outputElement.querySelector('textarea').value = '';
        }
        realtimeState.results = [];
        realtimeState.finalTranscript = '';
        updateStatus('🗑️ 結果をクリアしました');
    }

    window.downloadResults = function() {
        if (realtimeState.results.length === 0) {
            alert('ダウンロードする結果がありません。');
            return;
        }
        
        const text = realtimeState.results.join('\\n');
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `realtime_transcription_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ページロード後にグローバル登録（Gradio再レンダリング対策）
    window.addEventListener('load', () => {
        globalThis.startRealtimeRecognition = window.startRealtimeRecognition;
        globalThis.stopRealtimeRecognition  = window.stopRealtimeRecognition;
        globalThis.clearResults            = window.clearResults;
        globalThis.downloadResults         = window.downloadResults;
    });
    </script>
    """
    
    with gr.Blocks(title="音声文字起こしシステム", theme=gr.themes.Soft(), head=realtime_js) as interface:
        gr.Markdown("""
        # 🎤 音声文字起こしシステム
        
        NVIDIA Parakeet-TDT-CTCモデル（ファイル）またはGoogle Web Speech API（マイク）を使用した日本語音声文字起こしツールです。
        """)
        
        with gr.Tabs():
            # ファイルアップロードタブ
            with gr.TabItem("⬆️ ファイルから文字起こし"):
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
                
                # システム情報表示（動的更新）
                with gr.Row():
                    with gr.Column():
                        system_info_display = gr.Markdown(
                            value="### システム情報\n読み込み中...",
                            label="システム情報"
                        )
                        
                        # システム情報更新ボタン
                        refresh_btn = gr.Button("🔄 システム情報を更新", size="sm")

            # マイク入力タブ
            with gr.TabItem("🎙️ マイクから文字起こし"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mic_audio = gr.Audio(
                            label="マイク録音",
                            type="filepath",
                            sources=["microphone"]
                        )
                        mic_transcribe_btn = gr.Button(
                            "🎤 文字起こし実行",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        mic_output_text = gr.Textbox(
                            label="文字起こし結果",
                            lines=15,
                            max_lines=30,
                            placeholder="ここにマイクからの文字起こし結果が表示されます...",
                            show_copy_button=True
                        )

            # リアルタイム文字起こしタブ
            with gr.TabItem("⚡ リアルタイム文字起こし"):
                gr.Markdown("""
                ### 🎯 ブラウザネイティブ リアルタイム音声認識
                
                Web Speech APIによる音声検出とMediaRecorderによる高品質録音を組み合わせたハイブリッド文字起こし機能です。
                
                **特徴:**
                - 🌐 ブラウザネイティブ Web Speech API による音声検出
                - 🎤 MediaRecorder による高品質音声録音
                - ⚡ 音声検出イベントによる自動録音開始・終了
                - 🔄 Google Cloud Speech API による高精度文字起こし
                - 🎯 ハンズフリー操作（ボタン操作不要）
                
                **使い方:**
                1. 「開始」ボタンを押してマイク権限を許可
                2. 話しかけると自動的に録音・文字起こしが開始されます
                3. 無音状態になると自動的に録音停止・結果表示
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # コントロールボタン
                        with gr.Row():
                            realtime_start_btn = gr.Button(
                                "🎤 開始",
                                variant="primary",
                                size="lg",
                                elem_id="realtime_start_btn"
                            )
                            realtime_stop_btn = gr.Button(
                                "⏹️ 停止",
                                variant="secondary",
                                size="lg",
                                elem_id="realtime_stop_btn"
                            )
                        
                        # ステータス表示
                        realtime_status = gr.Textbox(
                            label="ステータス",
                            value="待機中 - 「開始」ボタンを押してください",
                            lines=3,
                            interactive=False,
                            elem_id="realtime_status"
                        )
                        
                        # 設定パネル
                        with gr.Accordion("⚙️ 詳細設定", open=False):
                            gr.Markdown("""
                            **音声認識設定:**
                            - 言語: 日本語 (ja-JP)
                            - 連続認識: 有効
                            - 中間結果: 有効
                            
                            **録音設定:**
                            - サンプルレート: 44.1kHz
                            - エンコーディング: WebM/MP3
                            - ノイズ除去: 有効
                            """)
                            
                            api_key_input = gr.Textbox(
                                label="Google Cloud Speech API Key (オプション)",
                                placeholder="独自のAPIキーを使用する場合は入力してください",
                                type="password",
                                elem_id="api_key_input"
                            )
                    
                    with gr.Column(scale=2):
                        # リアルタイム結果表示
                        realtime_output = gr.Textbox(
                            label="リアルタイム文字起こし結果",
                            lines=20,
                            max_lines=30,
                            placeholder="リアルタイム文字起こしを開始すると、ここに結果が表示されます...\n\n話しかけてください！",
                            show_copy_button=True,
                            autoscroll=True,
                            elem_id="realtime_output"
                        )
                        
                        # コントロールボタン
                        with gr.Row():
                            clear_results_btn = gr.Button(
                                "🗑️ 結果をクリア",
                                size="sm",
                                elem_id="clear_results_btn"
                            )
                            download_results_btn = gr.Button(
                                "💾 結果をダウンロード",
                                size="sm",
                                elem_id="download_results_btn"
                        )
        
        def update_system_info():
            """システム情報を更新"""
            info = get_system_info()
            
            markdown_text = f"""
### 📊 システム情報

#### 🖥️ デバイス情報
- **使用デバイス**: {info['device']}
- **CUDA利用可能**: {'✅ はい' if info['cuda_available'] else '❌ いいえ'}
- **GPU数**: {info['gpu_count']}
- **GPU名**: {info['gpu_name']}

#### 💾 メモリ情報
- **総GPU メモリ**: {info['total_memory']}
- **使用中メモリ**: {info['allocated_memory']}
- **キャッシュメモリ**: {info['cached_memory']}

#### 🤖 モデル情報
- **モデル**: nvidia/parakeet-tdt_ctc-0.6b-ja, silero-vad
- **モデル状態**: {'✅ 読み込み済み' if model_loaded else '⏳ 未読み込み'}
- **対応フォーマット**: WAV, MP3, FLAC, M4A など
"""
            return markdown_text
        
        # ブラウザネイティブ機能のため、Pythonイベントハンドラは不要
        # JavaScriptで直接処理されます
        
        # イベントハンドラ
        transcribe_btn.click(
            fn=process_audio_file,
            inputs=[audio_input],
            outputs=[output_text],
            show_progress=True
        )

        mic_transcribe_btn.click(
            fn=transcribe_mic_audio,
            inputs=[mic_audio],
            outputs=[mic_output_text],
            show_progress=True
        )
        
        # リアルタイム機能のJSイベントバインド
        realtime_start_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => startRealtimeRecognition()"
        )
        realtime_stop_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => stopRealtimeRecognition()"
        )
        clear_results_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => clearResults()"
        )
        download_results_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => downloadResults()"
        )
        
        # システム情報更新ハンドラ
        refresh_btn.click(
            fn=update_system_info,
            outputs=[system_info_display]
        )
        
        # 初期表示時にシステム情報を更新
        interface.load(
            fn=update_system_info,
            outputs=[system_info_display]
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