import os
import time
import datetime
import threading
import mido
from .config import CURRENT_WORKING_DIR, RECORDINGS_FOLDER, MIDI_TEMPO, MIDI_VELOCITY, MIDI_C4
class NoteRecorder:
    def __init__(self, framerate_hz, channels, tones, status_callback=None):
        self.midi_file_path = None  # Track the current MIDI file
        self.recording = False
        self.notes = []  # List of (key, sound, start_time, duration)
        self.midi_notes = []  # List of (tone, start_time, duration)
        self.current_notes = {}  # Maps key to (sound, start_time)
        self.current_midi_notes = {}  # Maps key to (tone, start_time)
        self.framerate_hz = framerate_hz
        self.channels = channels
        self.start_time = 0
        self.tones = tones  # Store the tone values for MIDI conversion
        self.status_callback = status_callback
        
        # Track recording segments
        self.segments = []  # List of (midi_path, start_time, end_time, status)
        self.current_segment_index = -1
        
        # Track generation in progress
        self.generation_in_progress = False
        self.current_continuation = None
        self.combined_midi_path = None
        
        # Ensure recordings directory exists
        self.recordings_dir = os.path.join(CURRENT_WORKING_DIR, RECORDINGS_FOLDER)
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Lazy load AI composer to prevent blocking UI
        self._ai_composer = None
        self._ai_composer_lock = threading.RLock()
        self._loading_model = False
    
    def start_recording(self):
        """Start recording a new segment"""
        self.recording = True
        self.notes = []
        self.midi_notes = []
        self.current_notes = {}
        self.current_midi_notes = {}
        self.start_time = time.time()
        
        if self.status_callback:
            self.status_callback("Recording started...", (255, 100, 100))
        
        print("Recording started...")
        
        # Preload the AI model in the background if not already loaded
        if self._ai_composer is None and not self._loading_model:
            _ = self.ai_composer  # This triggers the lazy loading

    # Add these methods to the NoteRecorder class in your paste-2.txt file

    def note_down(self, key, sound, key_index):
        """Record when a note starts playing"""
        if not self.recording:
            return
        
        # Get current time relative to recording start
        current_time = time.time() - self.start_time
        
        # Store the note start info
        self.current_notes[key] = (sound, current_time)
        
        # Also store MIDI note info
        tone = self.tones[key_index]
        self.current_midi_notes[key] = (tone, current_time)
        
        print(f"Note down: {key}, tone: {tone}, time: {current_time:.2f}")

    def note_up(self, key):
        """Record when a note stops playing"""
        if not self.recording or key not in self.current_notes:
            return
        
        # Get current time relative to recording start
        current_time = time.time() - self.start_time
        
        # Calculate duration
        sound, start_time = self.current_notes[key]
        duration = current_time - start_time
        
        # Add completed note to the notes list
        self.notes.append((key, sound, start_time, duration))
        
        # Also add to MIDI notes list
        if key in self.current_midi_notes:
            tone, midi_start_time = self.current_midi_notes[key]
            self.midi_notes.append((tone, midi_start_time, duration))
            
            print(f"Note up: {key}, tone: {tone}, duration: {duration:.2f}s")
        
        # Remove from current notes
        del self.current_notes[key]
        if key in self.current_midi_notes:
            del self.current_midi_notes[key]

    def stop_recording(self):
        """Stop the current recording"""
        if not self.recording:
            return
            
        # Ensure any active notes are properly ended
        current_time = time.time() - self.start_time
        
        # Close any notes that were still being held
        keys_to_remove = list(self.current_notes.keys())
        for key in keys_to_remove:
            sound, start_time = self.current_notes[key]
            duration = current_time - start_time
            
            # Add the completed note
            self.notes.append((key, sound, start_time, duration))
            
            # Also add to MIDI notes
            if key in self.current_midi_notes:
                tone, midi_start_time = self.current_midi_notes[key]
                self.midi_notes.append((tone, midi_start_time, duration))
        
        # Clear current notes
        self.current_notes = {}
        self.current_midi_notes = {}
        
        # Set recording state
        self.recording = False
        
        if self.status_callback:
            self.status_callback("Recording stopped", (100, 255, 100))
        
        print(f"Recording stopped. {len(self.notes)} notes recorded.")

    def save_midi_recording(self, anchor_note=None, output_path=None):
        """Save the recorded MIDI notes to a file"""
        if not self.midi_notes:
            if self.status_callback:
                self.status_callback("No notes to save", (255, 100, 100))
            return False
        
        # If no output path is specified, generate one
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.recordings_dir, f"recording_{timestamp}.mid")
        
        result = self._save_midi_to_path(output_path, anchor_note)
        
        if result and self.status_callback:
            self.status_callback(f"Saved to {os.path.basename(output_path)}", (100, 255, 100))
        
        return result
    
    def mark_segment(self):
        """Mark the current recording as a segment and start a new one"""
        if not self.recording:
            return False
            
        # First stop the current recording
        self.stop_recording()
        
        # Save the segment
        segment_path = self._save_segment()
        
        if segment_path:
            # Start a new recording segment
            self.start_recording()
            if self.status_callback:
                self.status_callback("New segment marked", (100, 200, 255))
            return True
        
        return False
    
    def _save_segment(self):
        """Save the current recording as a segment"""
        if not self.midi_notes:
            return None
            
        # Generate a filename for this segment
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        segment_index = len(self.segments)
        filename = f"segment_{segment_index}_{timestamp}.mid"
        segment_path = os.path.join(self.recordings_dir, filename)
        
        # Save MIDI file
        if self._save_midi_to_path(segment_path):
            # Add to segments list
            self.segments.append({
                "path": segment_path,
                "start_time": self.start_time,
                "end_time": time.time(),
                "status": "recorded",
                "continuation_path": None
            })
            
            # Update the current segment index
            self.current_segment_index = len(self.segments) - 1
            
            # Add to AI composer if available
            if self.ai_composer:
                try:
                    self.ai_composer.add_segment(segment_path, "recorded")
                except Exception as e:
                    print(f"Error adding segment to AI composer: {e}")
                    
            return segment_path
            
        return None
        
    def _save_midi_to_path(self, output_path, anchor_note=None):
        """Save MIDI notes to a specific path using sorted events to avoid negative delta times."""
        if not self.midi_notes:
            return False

        # Create a new MIDI file with one track
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=MIDI_TEMPO, time=0))

        # Determine reference note (defaults to middle C)
        reference_midi_note = MIDI_C4

        # Sort notes by start time
        sorted_notes = sorted(self.midi_notes, key=lambda x: x[1])

        # === Modified code: Collect all note on/off events ===
        events = []
        for tone, start_time, duration in sorted_notes:
            start_ticks = int(start_time * 1000)      # Convert to "ticks" (using ms here)
            duration_ticks = int(duration * 1000)       # Convert duration to "ticks"
            midi_note = reference_midi_note + tone

            # Ensure note is in valid MIDI range (0-127)
            if 0 <= midi_note <= 127:
                # Create note on and note off events with absolute times
                events.append((start_ticks, mido.Message('note_on', note=midi_note, velocity=MIDI_VELOCITY, time=0)))
                events.append((start_ticks + duration_ticks, mido.Message('note_off', note=midi_note, velocity=0, time=0)))

        # Sort events by absolute time
        events.sort(key=lambda x: x[0])

        # === Modified code: Compute delta times and add messages ===
        last_tick = 0
        for abs_time, message in events:
            delta = abs_time - last_tick
            # Guard against any potential negative delta time
            if delta < 0:
                delta = 0
            message.time = delta
            track.append(message)
            last_tick = abs_time

        # Save the MIDI file
        mid.save(output_path)
        self.midi_file_path = output_path
        return True
