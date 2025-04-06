import torch
import torch.nn as nn
from flash_attn import flash_attn_func
import sys
from flash_attn.modules.mha import MHA
import torch.functional as F

sys.path.append("MyLLM/CrispyModeling/Embedding")
from embedding import EmbeddingLayer    

class FlashAttentionBlockBase(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout= dropout

    def forward(self, q, k, v, causal=True):
        return flash_attn_func(q, k, v, causal=causal, dropout_p=self.dropout)

class FlashAttentionBlockMHA(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1, device="cuda", dtype="bfloat16"):
        super().__init__()
        
        self.device = device
        self.dtype = getattr(torch, dtype)
        
        self.attn = MHA(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            causal=True,
            device=self.device, 
            dtype=self.dtype,
            use_flash_attn=True,
        )

    def forward(self, x, cu_seqlens=None, max_seqlen=None):
        return self.attn(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

class AttentionBlock(nn.Module):

    def __init__(self, config, hidden_size=64, n_heads=16, dropout=0.1, max_seq_len=9048, device="cuda", dtype="bfloat16"):
        super().__init__()

        self.config = config
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.max_seq_len = max_seq_len  

        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // n_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, device=self.device, dtype=self.dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, device=self.device, dtype=self.dtype)
        self.ln = nn.LayerNorm(hidden_size, device=self.device, dtype=self.dtype)

      
        self.attn = FlashAttentionBlockMHA(hidden_size=hidden_size ,num_heads=self.n_heads, dropout=dropout) if self.config._attn_implementation == "flash_attention_2" else  nn.MultiheadAttention(embed_dim=hidden_size, num_heads=self.n_heads, batch_first=True, dropout=dropout, device=self.device, dtype=self.dtype)
        
        
        
    def forward(self, x, attention_mask=None):

        B, T, C = x.size()
        x = self.ln(x)
        
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        
    
        
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, T, n_heads, head_dim)
        
        if self.config._attn_implementation == "flash_attention_2":

            seq_lens = attention_mask.sum(dim=1)  # her örnekteki gerçek token sayıları, shape: (B,)
            cu_seqlens = F.pad(seq_lens.cumsum(0), (1, 0), value=0).to(torch.int32)  # shape: (B+1,)
            max_seqlen = seq_lens.max().item()  # batch içindeki en uzun dizi
            
            # x'i "packed" formata getir: (total, hidden_dim)
            x_flat = x.reshape(B * T, C).contiguous()

            out = self.attn(x_flat, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        else:
            q = q.reshape(B, T, self.n_heads * self.head_dim)  # [1, 5, 768]
            k = k.reshape(B, T, self.n_heads * self.head_dim)
            v = v.reshape(B, T, self.n_heads * self.head_dim)

            causal_mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).to(torch.bool).to(self.device)

            out, _ =  self.attn( q, k, v, attn_mask=causal_mask, key_padding_mask=attention_mask, need_weights=False)

        out = out.reshape(B, T, C)

        return self.o_proj(out)


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape=64, eps = 1e-05, elementwise_affine=True, bias=True, device="cuda", dtype="bfloat16"):
        super().__init__()

        self.device = device
        self.dtype = getattr(torch, dtype)

        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=self.dtype)

    def forward(self, x ):

        return self.ln(x)

class FeedforwardNetwork(nn.Module):

    def __init__(self, hidden_size=64, device="cuda", dtype="bfloat16"):
        super().__init__()

        self.device = device
        self.dtype = getattr(torch, dtype)

        self.ln1 = nn.Linear(hidden_size, 4 * hidden_size, device=self.device, dtype=self.dtype)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(4 * hidden_size, hidden_size, device=self.device,dtype=self.dtype)

    def forward(self, x):
        
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x



class DecoderBlock(nn.Module):

    def __init__(self, config, n_heads, hidden_size, max_seq_len, dtype="bfloat16", device="cuda"):
        super().__init__()

        self.config = config
        self.device = device
        self.dtype = getattr(torch, dtype)

        self.attention_block = AttentionBlock(config=self.config,hidden_size=hidden_size,n_heads=n_heads, max_seq_len=max_seq_len, device=device)
        self.feedforward_network = FeedforwardNetwork(hidden_size=hidden_size, device=device)
        self.layer_norm1 = LayerNorm(normalized_shape=hidden_size, device=device, dtype=dtype)
        self.layer_norm2 = LayerNorm(normalized_shape=hidden_size, device=device, dtype=dtype)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, attention_mask=None):
        # LayerNorm → Attention → Residual
        residual = x

        x = self.layer_norm1(x)

        x = residual + self.dropout(self.attention_block(x, attention_mask))

        # LayerNorm → Feedforward → Residual
        residual = x
        x = self.layer_norm2(x)

        x = residual + self.dropout(self.feedforward_network(x))

        return x
    
    

if __name__ == "__main__":

    device="cuda"

    input_ids = torch.tensor([[10, 20, 60, 45, 20]], dtype = torch.long, device=device)
    vocab_size = input_ids.max().item() + 1  # 60 + 1 = 61

    embedLayer = EmbeddingLayer(1000,hidden_size=768)

    input = embedLayer.forward(input_ids)

    decoder_block = DecoderBlock(hidden_size=768, device=device)
    
    print(decoder_block.forward(input))