import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiHeadSelfAttention(nn.Module):


    def __init__(self, d_model, num_heads):
        
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    

    def forward(self, x):

        B, T, D = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2] 

        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0) 

        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ V 
        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn_output)