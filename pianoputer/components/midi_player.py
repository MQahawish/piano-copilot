
import os
import sys
from pathlib import Path
from .config import CURRENT_WORKING_DIR, RECORDINGS_FOLDER
class MIDIPlayer:
    def __init__(self, status_callback=None):
        self.current_midi = None
        self.playing = False
        self.status_callback = status_callback
        self.process = None
        self.midi_directory = os.path.join(CURRENT_WORKING_DIR, RECORDINGS_FOLDER)
        # Ensure directory exists
        os.makedirs(self.midi_directory, exist_ok=True)
        
    def set_midi_directory(self, directory):
        """Set the directory to look for MIDI files"""
        if os.path.exists(directory) and os.path.isdir(directory):
            self.midi_directory = directory
            return True
        return False
        
    def load_midi(self, midi_path):
        """Load a MIDI file for playback"""
        if not os.path.exists(midi_path):
            if self.status_callback:
                self.status_callback(f"MIDI file not found: {midi_path}", (255, 100, 100))
            return False
            
        self.current_midi = midi_path
        if self.status_callback:
            self.status_callback(f"Loaded: {os.path.basename(midi_path)}", (100, 255, 100))
        return True
        
    def play(self):
        """Start playback of loaded MIDI file"""
        if not self.current_midi or not os.path.exists(self.current_midi):
            if self.status_callback:
                self.status_callback("No MIDI file loaded", (255, 100, 100))
            return False
            
        if self.playing:
            return True
            
        # Play MIDI using an external player (platform dependent)
        # This is a simple implementation - in production you might want to use
        # a proper MIDI library that integrates better with pygame
        try:
            import subprocess
            
            if sys.platform == 'win32':
                self.process = subprocess.Popen(['start', 'wmplayer', self.current_midi], 
                                               shell=True)
            elif sys.platform == 'darwin':
                self.process = subprocess.Popen(['afplay', self.current_midi])
            else:
                self.process = subprocess.Popen(['timidity', self.current_midi])
                
            self.playing = True
            
            if self.status_callback:
                self.status_callback(f"Playing {os.path.basename(self.current_midi)}", (100, 255, 100))
            return True
            
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Playback error: {str(e)}", (255, 100, 100))
            return False
            
    def pause(self):
        """Pause current playback"""
        if not self.playing or not self.process:
            return
            
        try:
            if self.process:
                self.process.terminate()
                self.process = None
                
            self.playing = False
            
            if self.status_callback:
                self.status_callback("Playback paused", (255, 200, 100))
                
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error pausing: {str(e)}", (255, 100, 100))
    
    def get_midi_files(self):
        """Return a list of MIDI files in the directory"""
        midi_files = []
        try:
            for file in os.listdir(self.midi_directory):
                if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                    full_path = os.path.join(self.midi_directory, file)
                    midi_files.append((file, full_path))
            return midi_files
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error reading MIDI directory: {str(e)}", (255, 100, 100))
            return []
