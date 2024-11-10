import rtmidi
import mido
import time

import rtmidi.midiutil

def play_midi(midi_file_path):
    # Create MIDI output
    midiout = rtmidi.MidiOut()
    
    # Check available ports and open the first one
    available_ports = midiout.get_ports()
    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My Virtual Port")
    
    # Load and play the MIDI file
    mid = mido.MidiFile(midi_file_path)
    
    for msg in mid.play():
        if not msg.is_meta:
            midiout.send_message(msg.bytes())
    
    # Clean up
    midiout.close_port()
    del midiout

# Use the function
play_midi("midis/inputurCv_pitch_corrected_basic_pitch.mid")