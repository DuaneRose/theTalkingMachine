
import queue 
import speech_recognition as sr
from speech_over import is_turn_complete
from pool import set_query
import mlx_whisper
import numpy as np
from tts import text_to_audio
from context_manager import clear_context
from print_color import print_yellow, print_red
from datetime import datetime

WHISPER_MODEL = "mlx-community/whisper-tiny.en-mlx"
user_words = ""
done = False
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
r = sr.Recognizer()
r.pause_threshold = 0.6
r.non_speaking_duration = 0.6
mic = sr.Microphone(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
mic_audio_q = queue.Queue() 
playback_q = queue.Queue() 
pause = False 

def calibrate_microphone():
    print("we need to calibrate the microphone first")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=3)
    print("Microphone calibrated. Starting audio capture...")

def capture_loop():
    calibrate_microphone()

    while not done:
        with mic as source:
            audio_data = r.listen(source, phrase_time_limit=7)
            mic_audio_q.put(audio_data.get_raw_data())

def warmup():
    print("Loading Whisper model...")
    mlx_whisper.transcribe(
        np.zeros(16000, dtype=np.float32),
        path_or_hf_repo=WHISPER_MODEL,
        language="en",
        fp16=False,
    )
    print("Whisper model ready.")

def analyze_text(text):
    global pause
    global user_words
    if text == " ":
        return
    elif "pause" in text.lower() and "conversation" in text.lower():
        print_red("\nPausing conversation. Say 'resume conversations' to continue.\n")
        text_to_audio("Pausing conversation. Say 'resume conversations' to continue.")
        pause = True
    elif "resume" in text.lower() and "conversation" in text.lower():
        print_red("\nResuming conversation.\n")
        text_to_audio("Resuming conversation.")
        pause = False
    elif "clear" in text.lower() and "context" in text.lower():
        print_red("\nClearing conversation context.\n")
        text_to_audio("Clearing conversation context.")
        clear_context()
    elif pause:
        print_red(f"\nConversation is paused. Ignoring input: {text}\n")
    else:
        user_words += text + ""
        likely_complete = is_turn_complete(user_words)
        print_yellow("\n" + text + f"({likely_complete:.2f})\n")
        if likely_complete > 0.6:
            print("\nsending to LLM...\n")
            set_query(user_words)
            user_words = ""

def transcribe_loop():
    global done
    global user_words

    while not done:
        if not mic_audio_q.empty():
            chunk = mic_audio_q.get()
            audio_data = sr.AudioData(chunk, mic.SAMPLE_RATE, mic.SAMPLE_WIDTH)
            try:
                text = transcribe(audio_data)
                analyze_text(text)
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

            if "exit" in text.lower():
                done = True
                print("Exiting...")

def transcribe(audio_data) -> str:
    raw = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
    audio_array = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    result = mlx_whisper.transcribe(
        audio_array,
        path_or_hf_repo=WHISPER_MODEL,
        language="en",
        fp16=False,
    )
    return result["text"].strip()