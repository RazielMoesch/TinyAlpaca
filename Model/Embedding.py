import torch
import torch.nn as nn



class Embedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_embed = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)

    
    def forward(self, x):


        batch_size, seq_len = x.shape

        token_emb = self.token_embed[x]
        pos_emb = self.pos_embed[:seq_len]
        pos_emb = pos_emb.unsqueeze(0)

        return token_emb + pos_emb