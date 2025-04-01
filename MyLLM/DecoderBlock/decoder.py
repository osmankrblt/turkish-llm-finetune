import torch
import torch.nn as nn
from flash_attn import flash_attn_func
import sys

sys.path.append("./MyLLM/Embedding/")
from embedding import EmbeddingLayer

class AttentionBlock(nn.Module):

    def __init__(self, embed_size=64, n_heads=16, dropout=0.2, use_flash_attention2 = True, device="cuda", dtype=torch.bfloat16):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads
        self.qkv_proj = nn.Linear(embed_size, 3 * embed_size, device=self.device, dtype=self.dtype)
        self.o_proj = nn.Linear(embed_size, embed_size, device=self.device, dtype=self.dtype)
        self.ln = nn.LayerNorm(embed_size, device=self.device, dtype=self.dtype)

        self.attn =  nn.MultiheadAttention(embed_dim=embed_size, num_heads=self.n_heads, batch_first=True, dropout=dropout, device=self.device, dtype=self.dtype)
        
        self.use_flash_attention2 = use_flash_attention2
        
    def forward(self, x, attn_mask=None):

        B, T, C = x.size()
        x = self.ln(x)
        
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, T, n_heads, head_dim)
        if self.use_flash_attention2:
            out = flash_attn_func(q, k, v, causal=True)
        else:
            out, _  = self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)

        out = out.reshape(B, T, C)

        return self.o_proj(out)


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape=64, eps = 1e-05, elementwise_affine=True, bias=True, device="cuda", dtype=torch.bfloat16):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=self.dtype)

    def forward(self, x ):

        return self.ln(x)

class FeedforwardNetwork(nn.Module):

    def __init__(self, embed_size=64, device="cuda", dtype=torch.bfloat16):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.ln1 = nn.Linear(embed_size, 4 * embed_size, device=self.device, dtype=self.dtype)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(4 * embed_size, embed_size, device=self.device,dtype=self.dtype)

    def forward(self, x):
        
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x



class DecoderBlock(nn.Module):

    def __init__(self, dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.attention_block = AttentionBlock(device=device)
        self.feedforward_network = FeedforwardNetwork(device=device)
        self.layer_norm1 = LayerNorm(device=device, dtype=self.dtype)
        self.layer_norm2 = LayerNorm(device=device, dtype=self.dtype)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, attn_mask=None):
        # LayerNorm → Attention → Residual
        residual = x

        x = self.layer_norm1(x)

        x = residual + self.dropout(self.attention_block(x, attn_mask))

        # LayerNorm → Feedforward → Residual
        residual = x
        x = self.layer_norm2(x)

        x = residual + self.dropout(self.feedforward_network(x))

        return x
    

if __name__ == "__main__":

    device="cuda"

    input_ids = torch.tensor([[10, 20, 60, 45, 20]], dtype = torch.long, device=device)
    vocab_size = input_ids.max().item() + 1  # 60 + 1 = 61

    embedLayer = EmbeddingLayer(1000,64)

    input = embedLayer.forward(input_ids)

    decoder_block = DecoderBlock(device=device)
    
    print(decoder_block.forward(input))