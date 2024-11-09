import librosa
import numpy as np
from collections import Counter
import psola
import soundfile as sf
import scipy.signal as sig
from pathlib import Path

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
    
    # Count unique notes
    note_counts = Counter(notes)
    unique_notes = set(notes)
    
    # Common scales with their note patterns
    scales = {
        'C Major': {'C', 'D', 'E', 'F', 'G', 'A', 'B'},
        'G Major': {'G', 'A', 'B', 'C', 'D', 'E', 'F#'},
        'D Major': {'D', 'E', 'F#', 'G', 'A', 'B', 'C#'},
        'A Minor': {'A', 'B', 'C', 'D', 'E', 'F', 'G'},
        'C Minor': {'C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'},
        'F Major': {'F', 'G', 'A', 'Bb', 'C', 'D', 'E'},
        'Bb Major': {'Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'},
        'Eb Major': {'Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'},
        'E Minor': {'E', 'F#', 'G', 'A', 'B', 'C', 'D'},
        'D Minor': {'D', 'E', 'F', 'G', 'A', 'Bb', 'C'},
        'G Minor': {'G', 'A', 'Bb', 'C', 'D', 'Eb', 'F'},
        'B Minor': {'B', 'C#', 'D', 'E', 'F#', 'G', 'A'},
        'F# Minor': {'F#', 'G#', 'A', 'B', 'C#', 'D', 'E'},
        'E Major': {'E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'}
    }
    
    # Find best matching scale
    best_match = None
    best_score = 0
    
    for scale_name, scale_notes in scales.items():
        score = len(unique_notes.intersection(scale_notes))
        if score > best_score:
            best_score = score
            best_match = scale_name
    
    return {
        'detected_scale': best_match,
        'notes_found': list(unique_notes),
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

if __name__ == "__main__":
    y, sr = librosa.load("ABC.wav")# y = time series, sr = sampling rate
    f0, voiced_flag, voiced_prob = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=1024)

    # Apply noise gate to the audio samples
    # y = apply_noise_gate(y, NOISE_GATE_THRESHOLD)

    # Perform the auto-tuning.
    # correction_function = closest_pitch if args.correction_method == 'closest' else \
    #     partial(aclosest_pitch_from_scale, scale=args.scale)
    #detect_scale(y,sr).detected_scale
    pitch_corrected_y = autotune(y, sr, aclosest_pitch_from_scale, "C:maj")
    filepath = Path("ABC.wav")


    # Write the corrected audio to an output file.
    filepath = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)
    sf.write(str(filepath), pitch_corrected_y, sr)

