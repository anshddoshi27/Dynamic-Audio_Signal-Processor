import pyaudio
import wave
import argparse
import sys
import random
import string

def record_audio(output_file, duration=5, sample_rate=44100, channels=1, chunk=1024):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Open stream
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk
    )
    
    print(f"Recording for {duration} seconds...")
    
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Recording finished!")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save the recorded data as a WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    print(f"Audio saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Record audio to a file')
    parser.add_argument('--duration', type=int, default=5, help='Recording duration in seconds (default: 5)')
    
    args = parser.parse_args()
    
    try:
        record_audio('inputs/input' + ''.join(random.choices(string.ascii_letters, k=4)) + '.wav', duration=args.duration)
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()