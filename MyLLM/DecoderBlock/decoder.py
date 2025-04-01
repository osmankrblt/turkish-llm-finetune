import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class AttentionBlock(nn.Module):

    def __init__(self, embed_size=64, n_heads=16, dropout=0.2, use_flash_attention2 = True):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads
        self.qkv_proj = nn.Linear(embed_size, 3 * embed_size)
        self.o_proj = nn.Linear(embed_size, embed_size)
        self.ln = nn.LayerNorm(embed_size)

        self.dropout=dropout
        self.embed_size = embed_size
        self.use_flash_attention2 = use_flash_attention2
        

            

    def forward(self, x):

        B, T, C = x.size()
        x = self.ln(x)
        
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, T, n_heads, head_dim)
        if self.use_flash_attention2:
            out = flash_attn_func(q, k, v, causal=True)
        else:
            out = nn.MultiheadAttention(embed_dim=self.embed_size, num_heads=self.n_heads, batch_first=True, dropout=self.dropout)

        out = out.view(B, T, C)

        return self.o_proj(out)


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

    def forward(self, x, attn_mask=None):
        # LayerNorm → Attention → Residual
        residual = x

        x = self.layer_norm(x)

        x = x + self.attention_block(x)

        x = self.layer_norm(x)

        x = x + self.feedforward_network(x)

        return x
    

if __name__ == "__main__":

    decoder_block = DecoderBlock()