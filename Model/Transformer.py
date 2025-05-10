import torch
import torch.nn as nn
from .DecoderLayer import DecoderLayer

class TinyAlpaca(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, vocab_size, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_ff, num_heads) for _ in range(num_layers)])
        self.final_linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        
        for layer in self.decoder_layers:
            x = layer(x)
            
        logits = self.final_linear(x)  # [batch_size, seq_len, vocab_size]
        return logits