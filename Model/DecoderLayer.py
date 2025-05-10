import torch
import torch.nn as nn
from .MaskedMultiHeadSelfAttention import  MaskedMultiHeadSelfAttention
from .FeedForwardNetwork import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.multi_head_attention = MaskedMultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output = self.multi_head_attention(x)
        attention_output = self.layer_norm1(x + attention_output)
        ffn_output = self.ffn(attention_output)
        output = self.layer_norm2(attention_output + ffn_output)
        
        return output
