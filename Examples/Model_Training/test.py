import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import torch
import os
from Tokenizer import BPETokenizer
from Model.Transformer import TinyAlpaca

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths (adjust as necessary)
VOCAB_PATH = "Examples/Model_Training/training_vocab.json"
MODEL_SAVE_PATH = "Examples/Model_Training/tinyalpaca.pth"

# Hyperparameters (must match training)
D_MODEL = 128
D_FF = 512
NUM_HEADS = 4
NUM_LAYERS = 4
MAX_SEQ_LEN = 128

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load_vocab(VOCAB_PATH)
vocab_size = len(tokenizer.vocab)

# Initialize model
model = TinyAlpaca(d_model=D_MODEL, d_ff=D_FF, num_heads=NUM_HEADS, 
                   vocab_size=vocab_size, num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN)

# Load trained model weights
if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
else:
    print(f"Model file not found at {MODEL_SAVE_PATH}")
    exit()

# Set model to device and evaluation mode
model.to(device)
model.eval()

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    """
    Generate text using the trained TinyAlpaca model.
    
    Args:
        model: The trained TinyAlpaca model.
        tokenizer: The BPETokenizer instance.
        prompt: The starting text prompt.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature (0 for greedy sampling).
    
    Returns:
        generated_text: The generated text as a string.
    """
    # Tokenize the prompt
    sequence = tokenizer.tokenize(prompt)
    if isinstance(sequence, str):
        sequence = [int(t) for t in sequence.split()]
    else:
        sequence = [int(t) for t in sequence]

    # Generate new tokens
    for _ in range(max_new_tokens):
        # Prepare input sequence (truncate to last max_seq_len tokens if necessary)
        if len(sequence) > MAX_SEQ_LEN:
            input_sequence = sequence[-MAX_SEQ_LEN:]
        else:
            input_sequence = sequence

        # Convert to tensor
        input_ids = torch.tensor([input_sequence], device=device)

        # Get model output
        with torch.no_grad():
            logits = model(input_ids)

        # Get logits for the next token
        next_logits = logits[0, -1, :]

        # Sample next token
        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = torch.argmax(next_logits).item()

        # Append to sequence
        sequence.append(next_token)

    # Decode the sequence back to text
    generated_text = tokenizer.detokenize(sequence)
    return generated_text

# Example usage
prompt = '''
Romeo oh
'''
generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7)
print("Generated Text:")
print(generated_text)