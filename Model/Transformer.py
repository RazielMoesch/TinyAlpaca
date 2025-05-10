import torch
import torch.nn as nn
from .Embedding import Embedding
from .DecoderLayer import DecoderLayer

class TinyAlpaca(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, vocab_size, num_layers, max_seq_len):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_ff, num_heads) for _ in range(num_layers)])
        self.final_linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        for layer in self.decoder_layers:
            x = layer(x)
            
        logits = self.final_linear(x)  # [batch_size, seq_len, vocab_size]
        return logits