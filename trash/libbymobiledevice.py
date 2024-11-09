from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import os
import time

class AudioTransferHandler(FileSystemEventHandler):
    def __init__(self, source_dir, destination_dir):
        self.source_dir = source_dir
        self.destination_dir = destination_dir

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.m4a', '.wav', '.mp3')):  # Add relevant audio formats
            filename = os.path.basename(event.src_path)
            dest_path = os.path.join(self.destination_dir, filename)
            
            # Use libimobiledevice to copy the file
            subprocess.run([
                'idevicefile',
                '-u', 'auto',  # automatically detect connected device
                'copy',
                event.src_path,
                dest_path
            ])
            print(f"Transferred: {filename}")

def start_monitoring(iphone_path, mac_destination):
    event_handler = AudioTransferHandler(iphone_path, mac_destination)
    observer = Observer()
    observer.schedule(event_handler, iphone_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Update these paths according to your setup
    IPHONE_RECORDINGS_PATH = "/var/mobile/Media/Recordings"  # typical path for voice memos
    MAC_DESTINATION = "/Users/romanpisani/ios_recordings"
    
    start_monitoring(IPHONE_RECORDINGS_PATH, MAC_DESTINATION)