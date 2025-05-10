import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import DataLoader
from Model.Transformer import TinyAlpaca
from Training.Dataset import TinyPacaDataset
from Training.train import train
from Tokenizer.tokenizer import BPETokenizer

# Keep original paths exactly as you had them
DATA_PATH = "Examples/Model_Training/TinyShakespeare.txt"
VOCAB_PATH = "Examples/Model_Training/training_vocab.json"
MODEL_SAVE_PATH = "Examples/Model_Training/tinyalpaca.pth"

EPOCHS = 50
BATCH_SIZE = 32
D_MODEL = 128
D_FF = 512
NUM_HEADS = 4
NUM_LAYERS = 4
LEARNING_RATE = 3e-4
MAX_SEQ_LEN = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BPETokenizer()

with open(DATA_PATH, "r") as f:
    text = f.read()

# Key fix: Only train tokenizer if vocab doesn't exist
if not os.path.exists(VOCAB_PATH):
    tokenizer.train(text, num_merges=150, vocab_path=VOCAB_PATH)
tokenizer.load_vocab(VOCAB_PATH)  # Always load to ensure consistency

vocab_size = len(tokenizer.vocab)
print(f"Vocab size: {vocab_size}, Max token ID: {max(tokenizer.vocab.keys())}")

# Key fix: Pass the already-trained tokenizer to prevent re-training
train_data = TinyPacaDataset(text, tokenizer, seq_len=MAX_SEQ_LEN, split="train", load_vocab_path=VOCAB_PATH)
val_data = TinyPacaDataset(text, tokenizer, seq_len=MAX_SEQ_LEN, split="val", load_vocab_path=VOCAB_PATH)

# Rest remains exactly the same...
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

model = TinyAlpaca(d_model=D_MODEL, d_ff=D_FF, num_heads=NUM_HEADS, 
                   vocab_size=vocab_size, num_layers=NUM_LAYERS, 
                   max_seq_len=MAX_SEQ_LEN).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=.00005)
loss_fn = torch.nn.CrossEntropyLoss()
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, 
#     T_0=1000,  
#     eta_min=1e-5
# )


train(epochs=EPOCHS, model=model, train_dl=train_loader, val_dl=val_loader, d_model=D_MODEL, d_ff=D_FF, 
      num_heads=NUM_HEADS, vocab_size=vocab_size, num_layers=NUM_LAYERS, 
      optimizer=optimizer, loss_fn=loss_fn, lr_scheduler=None, device=device)

torch.save(model.state_dict(), MODEL_SAVE_PATH)