from pydub import AudioSegment

def trim_wav(input_file):
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)
    
    # Extract first 10 seconds (10000 milliseconds)
    first_10_seconds = audio[:10000]
    
    # Export back to the same file
    first_10_seconds.export(input_file, format="wav")

# Example usage
if __name__ == "__main__":
    input_wav = "Mary.wav"
    trim_wav(input_wav)