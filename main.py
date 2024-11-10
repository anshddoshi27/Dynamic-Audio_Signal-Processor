import librosa
import numpy as np
from collections import Counter
import psola
import soundfile as sf
import scipy.signal as sig
from pathlib import Path
from basic_pitch.inference import predict, Model, predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)


# phone records audio file (wav)
# upload over bridge to project directory (somehow)
# upload triggers an event
# event runs main with the correct filename

SEMITONES_IN_OCTAVE = 12

# KEY:
"""
y = time series
sr = sampling rate
f0 = array of frequencies
voiced_flag = time series containing boolean flags indicating whether a frame is voiced or not
voiced_prob = time series containing the probability that a frame is voiced

pitches[..., f, t] contains instantaneous frequency at bin "f", time "t"
magnitudes[..., f, t] contains the corresponding magnitudes.

given : bpm
Noise gate threshold = 
"""

NOISE_GATE_THRESHOLD = 0.1  # Example threshold; adjust as needed

# Apply the noise gate to the audio samples and frequencies
def apply_noise_gate(samples, threshold):
    return np.where(np.abs(samples) < threshold, 0, samples)


def detect_scale(y, sr):
    
    # Extract pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Get the most prominent pitch at each time
    pitch_values = []
    for i in range(pitches.shape[1]): # .shape = dimensions of 2D np array, this range is # of columns
        index = magnitudes[:,i].argmax() # gets index of highest magnitude
        pitch = pitches[index,i]
        if pitch > 0:  # Filter out silence/noise
            pitch_values.append(pitch)
    
    # Convert frequencies to note names
    notes = []
    for pitch in pitch_values:
        note = librosa.hz_to_note(pitch)
        # Remove octave number to get just the note name
        note = ''.join([i for i in note if not i.isdigit()])
        notes.append(note)
    #print(notes)
    # Count unique notes
    note_counts = Counter(notes)
    top_notes = note_counts.most_common(8)
    
    # Common scales with their note patterns
    scales = {
        'C:maj': {'C', 'D', 'E', 'F', 'G', 'A', 'B'},
        'G:maj': {'G', 'A', 'B', 'C', 'D', 'E', 'F♯'},
        'D:maj': {'D', 'E', 'F♯', 'G', 'A', 'B', 'C♯'}, 
        'A:maj': {'A', 'B', 'C♯', 'D', 'E', 'F♯', 'G♯'},
        'E:maj': {'E', 'F♯', 'G♯', 'A', 'B', 'C♯', 'D♯'},
        'B:maj': {'B', 'C♯', 'D♯', 'E', 'F♯', 'G♯', 'A♯'},
        'F#:maj': {'F♯', 'G♯', 'A♯', 'B', 'C♯', 'D♯', 'E♯'},
        'C#:maj': {'C♯', 'D♯', 'E♯', 'F♯', 'G♯', 'A♯', 'B♯'},
        'F:maj': {'F', 'G', 'A', 'A♯', 'C', 'D', 'E'},
        'Bb:maj': {'A♯', 'C', 'D', 'D♯', 'F', 'G', 'A'},
        'Eb:maj': {'D♯', 'F', 'G', 'G♯', 'A♯', 'C', 'D'},
        'Ab:maj': {'G♯', 'A♯', 'C', 'C♯', 'D♯', 'F', 'G'},
        'Db:maj': {'C♯', 'D♯', 'F', 'F♯', 'G♯', 'A♯', 'C'},
        'Gb:maj': {'F♯', 'G♯', 'A♯', 'B', 'C♯', 'D♯', 'F'},
        'A:min': {'A', 'B', 'C', 'D', 'E', 'F', 'G'},
        'E:min': {'E', 'F♯', 'G', 'A', 'B', 'C', 'D'},
        'B:min': {'B', 'C♯', 'D', 'E', 'F♯', 'G', 'A'},
        'F#:min': {'F♯', 'G♯', 'A', 'B', 'C♯', 'D', 'E'},
        'C#:min': {'C♯', 'D♯', 'E', 'F♯', 'G♯', 'A', 'B'},
        'G#:min': {'G♯', 'A♯', 'B', 'C♯', 'D♯', 'E', 'F♯'},
        'D#:min': {'D♯', 'E♯', 'F♯', 'G♯', 'A♯', 'B', 'C♯'},
        'D:min': {'D', 'E', 'F', 'G', 'A', 'A♯', 'C'},
        'G:min': {'G', 'A', 'A♯', 'C', 'D', 'D♯', 'F'},
        'C:min': {'C', 'D', 'D♯', 'F', 'G', 'G♯', 'A♯'},
        'F:min': {'F', 'G', 'G♯', 'A♯', 'C', 'C♯', 'D♯'},
        'Bb:min': {'A♯', 'C', 'C♯', 'D♯', 'F', 'F♯', 'G♯'},
        'Eb:min': {'D♯', 'F', 'F♯', 'G♯', 'A♯', 'B', 'C♯'},
        'poop': {'C♯', 'A♯', 'A', 'D♯', 'D', 'B', 'G♯', 'E'}
    }
    best_match = None
    best_score = 0
    most_occurring = []
    for note in top_notes:
        most_occurring.append(note[0])
    most_occurring = set(most_occurring)

    print(top_notes)
    print(most_occurring)

    for scale_name, scale_notes in scales.items():
        score = len(most_occurring.intersection(scale_notes))
        if score > best_score:
            best_score = score
            best_match = scale_name
    
    
    return {
        'detected_scale': best_match,
        'notes_found': list(most_occurring),
        'note_frequencies': dict(note_counts)
    }

def closest_pitch(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""
    midi_note = np.around(librosa.hz_to_midi(f0))
    # To preserve the nan values.
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    # Convert back to Hz.
    return librosa.midi_to_hz(midi_note)


def degrees_from(scale: str):
    """Return the pitch classes (degrees) that correspond to the given scale"""
    degrees = librosa.key_to_degrees(scale)
    # To properly perform pitch rounding to the nearest degree from the scale, we need to repeat
    # the first degree raised by an octave. Otherwise, pitches slightly lower than the base degree
    # would be incorrectly assigned.
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    return degrees


def closest_pitch_from_scale(f0, scale):
    """Return the pitch closest to f0 that belongs to the given scale"""
    # Preserve nan.
    if np.isnan(f0):
        return np.nan
    degrees = degrees_from(scale)
    midi_note = librosa.hz_to_midi(f0)
    # Subtract the multiplicities of 12 so that we have the real-valued pitch class of the
    # input pitch.
    degree = midi_note % SEMITONES_IN_OCTAVE
    # Find the closest pitch class from the scale.
    degree_id = np.argmin(np.abs(degrees - degree))
    # Calculate the difference between the input pitch class and the desired pitch class.
    degree_difference = degree - degrees[degree_id]
    # Shift the input MIDI note number by the calculated difference.
    midi_note -= degree_difference
    # Convert to Hz.
    return librosa.midi_to_hz(midi_note)

def aclosest_pitch_from_scale(f0, scale):
    """Map each pitch in the f0 array to the closest pitch belonging to the given scale."""
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = closest_pitch_from_scale(f0[i], scale)
    # Perform median filtering to additionally smooth the corrected pitch.
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    # Remove the additional NaN values after median filtering.
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch


def autotune(audio, sr, correction_function, scale):
    # Set some basis parameters.
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Pitch tracking using the PYIN algorithm.
    f0, voiced_flag, voiced_probabilities = librosa.pyin(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
        sr=sr,
        fmin=fmin,
        fmax=fmax
    )

    # Apply the chosen adjustment strategy to the pitch.
    corrected_f0 = correction_function(f0, scale)

    # Pitch-shifting using the PSOLA algorithm.
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

def make_midi(wavfile):
    wavfile = "inputs/" + wavfile
    y, sr = librosa.load(wavfile)# y = time series, sr = sampling rate
    f0, voiced_flag, voiced_prob = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=1024)

    # Apply noise gate to the audio samples
    # y = apply_noise_gate(y, NOISE_GATE_THRESHOLD)

    # Perform the auto-tuning.
    # correction_function = closest_pitch if args.correction_method == 'closest' else \
    #     partial(aclosest_pitch_from_scale, scale=args.scale)
    print(detect_scale(y,sr)["detected_scale"])
    pitch_corrected_y = autotune(y, sr, aclosest_pitch_from_scale, detect_scale(y,sr)["detected_scale"])
    filepath = Path(wavfile)

    print()


    # Write the corrected audio to an output file.
    filepath = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)
    corrected_wav = wavfile[:-4] + "_pitch_corrected.wav"
    sf.write(str(filepath), pitch_corrected_y, sr)
    # model_output, midi_data, note_events = predict("ABC_pitch_corrected.wav")
    predict_and_save(
        [corrected_wav],
        "midis",
        True,
        False,
        False,
        False,
        basic_pitch_model
    )




