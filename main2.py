import librosa as lb
import numpy as np
import math
from scipy.signal import medfilt
import soundfile as sf

def remove_vibrato(y, sr, f0, voiced_flag, window_size=11):
    # 1. Smooth the f0 contour using median filtering
    smoothed_f0 = medfilt(f0, window_size)
    
    # 2. Create a time-varying frequency shift
    frequency_shift = f0 - smoothed_f0
    
    # 3. Phase vocoder to apply the frequency shift
    D = lb.stft(y)
    modified_D = lb.phase_vocoder(D=D, rate=1)
    
    # 4. Inverse STFT to get the modified audio
    y_modified = lb.istft(modified_D)
    
    return y_modified


y, sound = lb.load("ABC.wav")

# Estimate pitch and voicing
f0, voiced_flag, voiced_prob = lb.pyin(y, sr=sound, fmin=lb.note_to_hz('C2'), fmax=lb.note_to_hz('C7'), frame_length=1024)

y_without_vibrato = remove_vibrato(y, sound, f0, voiced_flag)

# Save the modified audio
with open("output_no_vibrato.wav", "wb") as f:
    sf.write(f, y_without_vibrato, sound)

# Print the pitch values
print(len(f0))
print(voiced_flag)
print(voiced_prob)

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

duration = lb.get_duration(y=y,sr = sound)
bpm = (32*60)/duration
print(len(y))

print(bpm)
print(duration)
freqs = []
for freq in f0:
    if not math.isnan(freq):
        print(freq_to_note(freq))
        freqs.append(freq_to_note(freq))


    # samples: [ A, A, A, A#, A, A#, A, A] -> vibrato
    # beats:     X        X          X 
seen_freqs = set()
for freq in freqs:
    if freq not in seen_freqs:
        seen_freqs.add(freq)
        print(freqs.count(freq))