import soundfile as sf
import scipy.signal
import numpy as np
from scipy.fft import fft
import math

# def freq_to_note(freq):
#     if freq == 0:
#         return None
#     print("freq: " + str(freq))
#     notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
#     note_number = 12 * math.log2(freq/440) + 49
#     print("number: " + str(note_number))
#     note_number = round(note_number)
#     note = notes[note_number % 12]
#     octave = (note_number + 8) // 12
#     print("note:" + note)
#     return f"{note}{octave}"
def freq_to_note(frequency):
    # define constants that control the algorithm
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] # these are the 12 notes in each octave
    OCTAVE_MULTIPLIER = 2 # going up an octave multiplies by 2
    KNOWN_NOTE_NAME, KNOWN_NOTE_OCTAVE, KNOWN_NOTE_FREQUENCY = ('A', 4, 440) # A4 = 440 Hz

    # calculate the distance to the known note
    # since notes are spread evenly, going up a note will multiply by a constant
    # so we can use log to know how many times a frequency was multiplied to get from the known note to our note
    # this will give a positive integer value for notes higher than the known note, and a negative value for notes lower than it (and zero for the same note)
    note_multiplier = OCTAVE_MULTIPLIER**(1/len(NOTES))
    frequency_relative_to_known_note = frequency / KNOWN_NOTE_FREQUENCY
    distance_from_known_note = math.log(frequency_relative_to_known_note, note_multiplier)

    # round to make up for floating point inaccuracies
    distance_from_known_note = round(distance_from_known_note)

    # using the distance in notes and the octave and name of the known note,
    # we can calculate the octave and name of our note
    # NOTE: the "absolute index" doesn't have any actual meaning, since it doesn't care what its zero point is. it is just useful for calculation
    known_note_index_in_octave = NOTES.index(KNOWN_NOTE_NAME)
    known_note_absolute_index = KNOWN_NOTE_OCTAVE * len(NOTES) + known_note_index_in_octave
    note_absolute_index = known_note_absolute_index + distance_from_known_note
    note_octave, note_index_in_octave = note_absolute_index // len(NOTES), note_absolute_index % len(NOTES)
    note_name = NOTES[note_index_in_octave]
    return (note_name, note_octave)


def extract_piano_notes(audio_path):
    # Load audio file
    audio_data, sample_rate = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Parameters for analysis
    window_size = 4096  # Increased for better frequency resolution
    hop_length = 1024   # Increased to reduce overlapping detections
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    # Detect onsets with higher threshold
    onset_envelope = np.abs(audio_data)
    onset_envelope = scipy.signal.medfilt(onset_envelope, 11)
    peaks = scipy.signal.find_peaks(onset_envelope, 
                                  distance=hop_length * 2,  # Increased minimum distance
                                  height=0.1,              # Adjusted threshold
                                  prominence=0.1)[0]       # Added prominence requirement
    
    notes = []
    last_note = None
    min_time_between_notes = 0.1  # Minimum time between different notes
    
    for peak in peaks:
        time = peak / sample_rate
        
        # Extract the window of audio after the onset
        start = peak
        end = min(start + window_size, len(audio_data))
        window = audio_data[start:end]
        
        if len(window) < window_size:
            window = np.pad(window, (0, window_size - len(window)))
        
        # Apply Hanning window
        window = window * np.hanning(len(window))
        
        # Compute FFT
        spectrum = abs(fft(window))
        spectrum = spectrum[:len(spectrum)//2]
        
        # Find the frequency with maximum amplitude
        freq_bins = np.fft.fftfreq(window_size, 1/sample_rate)
        freq_bins = freq_bins[:len(freq_bins)//2]
        max_freq = freq_bins[np.argmax(spectrum)]
        
        # Convert frequency to note
        note = freq_to_note(abs(max_freq))
        
        # Only add note if it's different from the last note or enough time has passed
        if note and (last_note is None or 
                    (note != last_note[1] and time - last_note[0] >= min_time_between_notes)):
            notes.append({
                'time': time,
                'note': note
            })
            last_note = (time, note)
            
    return notes

# Example usage
audio_file = "ABC.wav"
detected_notes = extract_piano_notes(audio_file)

# Print results
for note in detected_notes:
    print(f"Time: {note['time']:.2f}s, Note: {note['note']}")