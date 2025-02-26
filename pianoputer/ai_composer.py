# Create a new file: pianoputer/ai_composer.py
import sys
import os
import torch
from pathlib import Path
from pianoputer.model.encoding_decoding import Encoder, Decoder
# Import your model components
from pianoputer.model.sample import load_model, SimpleTokenizer, generate_music_sequence

class AIComposer:
    def __init__(self, model_path="runs/3layers12heads32batch512seq-simple", 
                 tokenizer_path="music/midi-processor/tokenizers/simple_tokenizer.json"):
        # Initialize model and tokenizer
        self.model = load_model(model_path)
        self.tokenizer = SimpleTokenizer(tokenizer_path)
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def process_user_recording(self, midi_path):
        """Convert user MIDI to model input format"""
        # Process MIDI to text representation
        return self.encoder.midi_to_text(midi_path)
        
    def generate_continuation(self, encoded_input, temperature=0.9, top_k=100, max_new_tokens=256):
        """Generate continuation based on user input"""
        return generate_music_sequence(
            self.model, 
            self.tokenizer,
            encoded_input,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens
        )
        
    def create_combined_midi(self, original_text, generated_text, output_path):
        """Combine user and AI parts into a single MIDI"""
        # Remove any duplicate START tokens and join
        # (Implementation details depend on your exact encoding format)
        combined_text = self._merge_text_sequences(original_text, generated_text)
        self.decoder.text_to_midi(combined_text, os.path.dirname(output_path), 
                                 os.path.basename(output_path).replace('.mid', ''))
        return output_path