import torch
from contextlib import nullcontext
import random
seed = random.randint(0, 1000000)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def generate_music_sequence_stream(model, tokenizer, initial_prompt, temperature, top_k, max_new_tokens):
    """
    Generate a music sequence using a sliding window approach that starts by accumulating tokens up to 512,
    then slides the context window by 1 token at a time.
    """
    # Prepare initial sequence
    init_ids = tokenizer.encode(initial_prompt, add_special_tokens=True)
    full_sequence = init_ids.copy()
    
    print(f"Starting generation with {len(full_sequence)} initial tokens")
    print(f"Will generate exactly {max_new_tokens} tokens")
    
    # Stream the initial tokens
    initial_text = tokenizer.decode(full_sequence, skip_special_tokens=False)
    yield initial_text, initial_text
    
    # Generate tokens one by one
    tokens_generated = 0
    context_window = 512  # Maximum context window size

    while tokens_generated < max_new_tokens:
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
