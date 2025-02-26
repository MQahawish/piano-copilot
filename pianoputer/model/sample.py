import os
import sys
import torch
import random
from glob import glob
from pathlib import Path
from contextlib import nullcontext
# Custom tokenizer instead of HuggingFace
import json
from model import GPTConfig, GPT
# Remove import for zip_unzip since we don't need it anymore
from encoding_decoding import Decoder
# from midi_decoder import CompactDecoder

# -----------------------------------------------------------------------------
# Configuration
init_from = 'resume'
out_dir = os.path.join('runs', '3layers12heads32batch512seq-simple')
tokenizer_path = os.path.join('music', 'midi-processor', 'tokenizers', 'simple_tokenizer.json')
num_samples = 5  # Number of different samples to generate
tokens_to_sample = 5  # Number of initial tokens to sample from each file

# Global generation parameters
temperature = 0.9
# ===== Added Repetition Penalty =====
repetition_penalty = 1.2  # you can adjust this value
top_k = 100
max_new_tokens = 512      # For non-streaming generation
fixed_tokens = 512        # For streaming generation

seed = random.randint(0, 1000000)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
compile = False

# Paths for encoded files
encoding_paths = [
    r"C:\Users\97250\OneDrive\Documents\GitHub\GPT-R0\encodings",
    r"C:\Users\97250\OneDrive\Documents\GitHub\GPT-R0\augmented_encodings\encodings"
]

# Create directories if they don't exist
samples_dir = "samples"
midi_samples_dir = "midi-samples"  # Direct MIDI output folder
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(midi_samples_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Setup CUDA and device settings
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_model():
    """Load the GPT model from checkpoint"""
    print(f"Loading model from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)
    return model

class SimpleTokenizer:
    """Custom tokenizer for the simple format with d/n/v tokens"""
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.tokenizer_data = json.load(f)
        
        # Convert string keys to int for itos if needed
        self.itos = {int(k): v for k, v in self.tokenizer_data['itos'].items()}
        self.stoi = self.tokenizer_data['stoi']
        self.vocab_size = self.tokenizer_data['vocab_size']
        
        # Special tokens
        self.start_token = "START"
        self.pad_token = "[PAD]"
        self.start_token_id = self.stoi.get(self.start_token, 0)
        
        # Get PAD token ID (usually the last one)
        if self.pad_token in self.stoi:
            self.pad_token_id = self.stoi[self.pad_token]
        else:
            self.pad_token_id = self.vocab_size - 1
    
    def encode(self, text, add_special_tokens=False):
        """Convert text to token ids"""
        # Split the text by spaces to get individual tokens
        tokens = text.split()
        
        # Convert to token IDs, using pad_token_id for unknown tokens
        token_ids = []
        for token in tokens:
            if token in self.stoi:
                token_ids.append(self.stoi[token])
            else:
                token_ids.append(self.pad_token_id)
                print(f"Warning: unknown token '{token}' found in text")
        
        # Add start token if requested
        if add_special_tokens:
            token_ids = [self.start_token_id] + token_ids
            
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=False):
        """Convert token ids back to text"""
        tokens = []
        for idx in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens:
                if idx == self.start_token_id or idx == self.pad_token_id:
                    continue
            
            # Convert id to token string
            if idx in self.itos:
                token = self.itos[idx]
                tokens.append(token)
            else:
                tokens.append(self.pad_token)
                print(f"Warning: unknown token ID {idx} encountered during decoding")
        
        # Join tokens with spaces for the simple tokenizer format
        return " ".join(tokens)

def load_tokenizer():
    """Load the custom simple tokenizer"""
    print(f"Loading simple tokenizer from {tokenizer_path}")
    return SimpleTokenizer(tokenizer_path)

def get_random_encoding():
    """
    Randomly select a txt file from one of the encoding paths and extract initial tokens,
    searching recursively through subdirectories
    """
    # Randomly choose a path
    chosen_path = random.choice(encoding_paths)
    
    # Get all txt files in the chosen path and its subdirectories
    txt_files = []
    for root, dirs, files in os.walk(chosen_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    if not txt_files:
        raise ValueError(f"No txt files found in {chosen_path} or its subdirectories")
    
    # Randomly choose a file
    chosen_file = random.choice(txt_files)
    
    # Get relative path for filename
    rel_path = os.path.relpath(chosen_file, chosen_path)
    file_id = Path(rel_path).with_suffix('').as_posix().replace('/', '_')
    
    # Read the file and get initial tokens
    with open(chosen_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        # Split by spaces and take first n tokens
        tokens = content.split()[:tokens_to_sample]
        return " ".join(tokens), file_id

def generate_music_sequence(model, tokenizer, initial_prompt, temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens):
    """
    Generate a single complete music sequence without using BOS/EOS tokens
    """
    # Prepare initial sequence directly from the prompt
    init_ids = tokenizer.encode(initial_prompt, add_special_tokens=True)
    current_ids = torch.tensor([init_ids], dtype=torch.long).to(device)
    
    full_sequence = []
    full_sequence.extend(current_ids[0].tolist())
    
    print(f"Starting generation with {len(full_sequence)} tokens")
    
    # Generate until we reach max length
    while len(full_sequence) < max_new_tokens:
        # Take last 1022 tokens as context
        context = full_sequence[-1022:] if len(full_sequence) > 1022 else full_sequence
        input_ids = torch.tensor([context], dtype=torch.long).to(device)
        
        # Generate next token
        with torch.no_grad():
            with ctx:
                logits, _ = model(input_ids)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
        
        next_token_id = next_token.item()
        full_sequence.append(next_token_id)
    
    return tokenizer.decode(full_sequence, skip_special_tokens=False)

def get_initial_sequence(tokenizer, custom_mode=False, tokens_to_sample=tokens_to_sample):
    """
    Get the initial sequence either through random sampling or custom mode
    
    Args:
        tokenizer: The tokenizer to use for proper tokenization
        custom_mode (bool): If True, returns a sequence of <I0> tokens
        tokens_to_sample (int): Number of tokens to sample if in sampling mode
    
    Returns:
        tuple: (initial_sequence, file_source)
    """
    if custom_mode:
        return f"S", "custom_init"
    else:
        # Randomly choose a path
        chosen_path = random.choice(encoding_paths)
        
        # Get all txt files in the chosen path and its subdirectories
        txt_files = []
        for root, dirs, files in os.walk(chosen_path):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        
        if not txt_files:
            raise ValueError(f"No txt files found in {chosen_path} or its subdirectories")
        
        # Randomly choose a file
        chosen_file = random.choice(txt_files)
        
        # Get relative path for filename
        rel_path = os.path.relpath(chosen_file, chosen_path)
        file_id = Path(rel_path).with_suffix('').as_posix().replace('/', '_')
        
        # Read the file and properly tokenize it
        with open(chosen_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Properly tokenize the content
            token_ids = tokenizer.encode(content, add_special_tokens=False)
            # Take first n tokens
            initial_tokens = token_ids[:tokens_to_sample]
            # Decode back to text
            initial_sequence = tokenizer.decode(initial_tokens, skip_special_tokens=False)
            return initial_sequence, file_id
        
def generate_music_sequence_stream(model, tokenizer, initial_prompt, temperature=temperature, top_k=top_k, fixed_tokens=fixed_tokens):
    """
    Generate a music sequence using a sliding window approach that starts by accumulating tokens up to 512,
    then slides the context window by 1 token at a time.
    """
    # Prepare initial sequence
    init_ids = tokenizer.encode(initial_prompt, add_special_tokens=True)
    full_sequence = init_ids.copy()
    
    print(f"Starting generation with {len(full_sequence)} initial tokens")
    print(f"Will generate exactly {fixed_tokens} tokens")
    
    # Stream the initial tokens
    initial_text = tokenizer.decode(full_sequence, skip_special_tokens=False)
    yield initial_text, initial_text
    
    # Generate tokens one by one
    tokens_generated = 0
    context_window = 512  # Maximum context window size

    while tokens_generated < fixed_tokens:
        # Use the entire sequence until we reach the context window,
        # then slide by one token at a time.
        if len(full_sequence) < context_window:
            context = full_sequence
        else:
            context = full_sequence[-(context_window - 1):]
        
        input_ids = torch.tensor([context], dtype=torch.long).to(device)
        
        with torch.no_grad():
            with ctx:
                logits, _ = model(input_ids)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
        
        next_token_id = next_token.item()
        full_sequence.append(next_token_id)
        tokens_generated += 1
        
        new_token = tokenizer.decode([next_token_id], skip_special_tokens=False)
        full_text = tokenizer.decode(full_sequence, skip_special_tokens=False)
        yield new_token, full_text

def generate_and_save_midi_streaming(model, tokenizer, input_text, output_midi_path, 
                                    temperature=temperature, top_k=top_k, fixed_tokens=fixed_tokens,
                                    time_sig="4/4", key_info=(0, 0), tempo=120.0):
    """
    Generate a music sequence and directly convert to MIDI (no unzipping needed for simple tokenizer)
    """
    full_sequence = ""
    
    # Stream the generation
    for new_token, full_text in generate_music_sequence_stream(
        model, tokenizer, input_text,
        temperature=temperature,
        top_k=top_k,
        fixed_tokens=fixed_tokens
    ):
        full_sequence = full_text
        print(new_token, end='', flush=True)
    
    print("\nGeneration complete!")
    
    # For simple tokenizer, we should keep the tokens separated by spaces
    # Just trim any excessive whitespace
    clean_sequence = " ".join(full_sequence.split())
    
    # Save the text sequence
    with open(output_midi_path.replace('.mid', '.txt'), 'w', encoding='utf-8') as f:
        f.write(clean_sequence)
    
    # Create an instance of the Decoder class
    decoder = Decoder()
    
    # Convert the sequence directly to MIDI (no unzipping needed for simple tokenizer)
    # Pass the tokens as they are to the decoder instance
    decoder.text_to_midi(
        text=clean_sequence,
        output_dir=os.path.dirname(output_midi_path),
        name=os.path.basename(output_midi_path).replace('.mid', ''),
        bpm=tempo
    )
    
    return clean_sequence

def main(custom_mode=False, num_samples=5, tokens_to_sample=tokens_to_sample):
    """
    Main function that directly converts generated sequences to MIDI files and 
    continues numbering from existing files
    """
    # Load model and tokenizer
    model = load_model()
    tokenizer = load_tokenizer()
    
    # Count existing files in output directories
    existing_text_files = len([f for f in os.listdir(samples_dir) if f.endswith('.txt')])
    existing_midi_files = len([f for f in os.listdir(midi_samples_dir) if f.endswith('.mid')])
    
    # Use the maximum count to ensure we don't overwrite any files
    start_index = max(existing_text_files, existing_midi_files)
    print(f"Found {existing_text_files} text files and {existing_midi_files} MIDI files")
    print(f"Starting generation from index {start_index + 1}")
    
    # Generate multiple samples
    for i in range(num_samples):
        # Calculate the current file index
        current_index = start_index + i + 1
        
        # Get initial sequence based on mode
        initial_sequence, file_source = get_initial_sequence(
            tokenizer=tokenizer,
            custom_mode=custom_mode,
            tokens_to_sample=tokens_to_sample
        )
        
        print(f"\nGenerating sample {current_index} using sequence from {file_source}")
        print(f"Initial sequence: {initial_sequence}")
        
        try:
            # Define output paths with the updated index
            text_output_path = os.path.join(samples_dir, f"sample_{current_index}.txt")
            midi_output_path = os.path.join(midi_samples_dir, f"sample_{current_index}.mid")
            
            # Generate sequence and convert directly to MIDI
            generated_sequence = generate_and_save_midi_streaming(
                model, 
                tokenizer, 
                initial_sequence,
                output_midi_path=midi_output_path,
                tempo=120.0  # Default tempo
            )
            
            print(f"\nGenerated sequence saved as: {text_output_path}")
            print(f"MIDI file created at: {midi_output_path}")
            
        except Exception as e:
            print(f"\nError generating sample {current_index}: {str(e)}")
            continue

if __name__ == "__main__":
    CUSTOM_MODE = False  # Set to True to use <I0> initialization
    NUM_SAMPLES = 5
    TOKENS_TO_SAMPLE = 30
    
    try:
        # Run the main function which now directly converts to MIDI
        main(
            custom_mode=CUSTOM_MODE,
            num_samples=NUM_SAMPLES,
            tokens_to_sample=TOKENS_TO_SAMPLE
        )
        # No need to call process_all_samples or convert_samples_to_midi
        print("Generation and MIDI conversion complete!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise