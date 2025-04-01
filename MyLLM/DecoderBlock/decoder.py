import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class AttentionBlock(nn.Module):

    def __init__(self, use_flash_attention2 = True):
        super().__init__()

        if use_flash_attention2:

            self.attention = flash_attn_func.FlashAttention(dim=64, num_heads=8, dropout=0.1)


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape=64, eps = 1e-05, elementwise_affine=True, bias=True, device="cuda", dtype=None):
        super().__init__()

        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype)

    def forward(self, x ):

        return self.ln(x)

class FeedforwardNetwork(nn.Module):

    def __init__(self, embed_size=64):
        super().__init__()

        self.ln1 = nn.Linear(embed_size, 4 * embed_size),
        self.gelu = nn.GELU(),
        self.ln2 = nn.Linear(4 * embed_size, embed_size)

    def forward(self, x):
        
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x



class DecoderBlock(nn.Module):

    def __init__(self, ):
        super().__init__()

        self.attention_block = AttentionBlock()
        self.feedforward_network = FeedforwardNetwork()
        self.layer_norm = LayerNorm()

    def forward(self, x, encoder_output):

        x = self.layer_norm(x)

        x = x + self.attention_block(x)

        x = self.layer_norm(x)

        x = x + self.feedforward_network(x)

        return x
    

if __name__ == "__main__":

    flash_attn_func.FlashAttention(dim=64, num_heads=8, dropout=0.1)