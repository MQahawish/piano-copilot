
import json
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
