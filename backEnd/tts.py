import queue, numpy as np, sounddevice as sd, time
from piper import PiperVoice
import threading

PIPER_MODEL = "/Users/duanegennaro/piper-voices/en_US-kathleen-low.onnx"
mouth = PiperVoice.load(PIPER_MODEL)
PIPER_SAMPLE_RATE = mouth.config.sample_rate
print(f"Piper loaded. Sample rate: {PIPER_SAMPLE_RATE}")

playback_q: "queue.Queue[np.ndarray | None]" = queue.Queue()
audio_stream = sd.OutputStream(samplerate=PIPER_SAMPLE_RATE, channels=1, dtype="float32")
audio_stream.start()

def playback_worker():
    while True:
        chunk = playback_q.get()
        if chunk is None:
            playback_q.task_done()
            break
        audio_stream.write(chunk.reshape(-1, 1))
        playback_q.task_done()

threading.Thread(target=playback_worker, daemon=True).start()

def text_to_audio(text: str, play: bool = True):
    for audio_chunk in mouth.synthesize(text):
        audio = audio_chunk.audio_float_array  # already float32, no conversion needed
        if play:
            playback_q.put(audio)