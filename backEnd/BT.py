import threading, queue 
import speech_recognition as sr
from tts import text_to_audio
from stt import capture_loop, transcribe_loop, warmup
from pool import get_query, delete_query
from brain import communicate
from print_color import print_green

done = False

text_q = queue.Queue()

def comsunicate_with_llm():
    global text_q
    while not done:
        if get_query() != "":
            current_query = get_query()
            delete_query()

            buffer = ""

            for token in communicate(current_query):
                print_green(token)
                buffer += token

                if buffer.endswith((".", "?", "!", "\n")):
                    text_q.put(buffer.strip())
                    buffer = ""

            if buffer.strip():
                text_q.put(buffer.strip())

def tts_worker():
    global done
    while not done:
        if not text_q.empty():
            text = text_q.get()
            text_to_audio(text, play=True)
            
def main():
    print("staring up porgram please wait.")
    warmup()
    capture_thread = threading.Thread(target=capture_loop)
    transcribe_thread = threading.Thread(target=transcribe_loop)
    communicate_thread = threading.Thread(target=comsunicate_with_llm)
    tts_thread = threading.Thread(target=tts_worker)
    capture_thread.start()
    transcribe_thread.start()
    communicate_thread.start()
    tts_thread.start()
    print("please don't speak until spoken too.")
    text_to_audio("Hello, I am your AI assistant. How can I help you today?")

if __name__ == "__main__":
    main()