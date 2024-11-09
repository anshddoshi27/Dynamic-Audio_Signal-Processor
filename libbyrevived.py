from main4 import make_midi
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import os
import time

class AudioTransferHandler(FileSystemEventHandler):
    def __init__(self, source_dir):
        self.source_dir = source_dir

    def on_created(self, event):
        filename = os.path.basename(event.src_path)
        print("name " + filename)
        print("truncated: " + filename[-10:])

        if(filename[-10:] == "rected.wav"):
            return
        make_midi(filename)
        print(f"roman pooped: {filename}")

def start_monitoring(recordings_path):

    event_handler = AudioTransferHandler(recordings_path)
    observer = Observer()
    observer.schedule(event_handler, recordings_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    RECORDINGS_PATH = "inputs"
    
    start_monitoring(RECORDINGS_PATH)