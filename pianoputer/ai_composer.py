# ===== NEW FILE: ai_composer.py =====
import os
import torch
import json
from model.model import GPTConfig, GPT
from model.encoding_decoding import Encoder, Decoder  # Your existing module
from model.simple_tokenizer import SimpleTokenizer  # Adjust the import as needed
from model.generate import generate_music_sequence_stream  # Your model generation function

class AIComposer:
    def __init__(self, model_checkpoint_dir, tokenizer_path, device=None, compile_model=False):
        self.model = self.load_model(model_checkpoint_dir, device, compile_model)
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.encoder = Encoder()   # Use your existing Encoder
        self.decoder = Decoder()   # Use your existing Decoder
        
    def load_model(self, model_checkpoint_dir, device=None, compile_model=False):
        ckpt_path = os.path.join(model_checkpoint_dir, 'checkpoint.pt')
        checkpoint = torch.load(ckpt_path, map_location=device or 'cpu')
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device or 'cpu')
        if compile_model:
            model = torch.compile(model)
        return model

    def load_tokenizer(self, tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        return SimpleTokenizer(tokenizer_path)

    def process_user_recording(self, midi_path):
        """
        Convert the recorded MIDI file to a text encoding using your existing Encoder.
        """
        return self.encoder.midi_to_text(midi_path)

    def generate_continuation(self, encoded_input, temperature=0.9, top_k=100, max_new_tokens=256):
        """
        Generate continuation tokens using your model's sampling routine.
        """
        full_sequence = ""
    
        # Stream the generation
        for new_token, full_text in generate_music_sequence_stream(
            self.model,
            self.tokenizer,
            encoded_input,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens
        ):
            full_sequence = full_text
        return full_sequence
    
    def create_combined_midi(self, encoded_input, generated_output, output_path):
        """
        Concatenate the original encoding with the generated tokens and convert it to a MIDI file.
        """
        combined_encoding = encoded_input + " " + generated_output
        self.decoder.text_to_midi(
            text=combined_encoding,
            output_dir=os.path.dirname(output_path),
            name=os.path.basename(output_path).replace('.mid', ''),
            bpm=120.0  # Adjust BPM if needed
        )
