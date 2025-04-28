import torch
import torch.nn as nn
from flash_attn import flash_attn_func
import sys
from flash_attn.modules.mha import MHA
import torch.nn.functional as F
from typing import Optional, OrderedDict
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, GenerationMixin
from transformers import AutoConfig
from transformers.modeling_utils import load_state_dict
from safetensors.torch import load_file as safe_load_file
import os
import torch.nn as nn
import torch.nn.init as init


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out, device="cuda", dtype="bfloat16"):
        super().__init__()
        self.device = device
        self.dtype = getattr(torch, dtype)

        self.linear1 = nn.Linear(dim_in, dim_out * 2, device=self.device, dtype=self.dtype)
        self.linear2 = nn.Linear(dim_out, dim_in, device=self.device, dtype=self.dtype)

    def forward(self, x):
        x = x.to(self.device).to(self.dtype)

        x_proj = self.linear1(x)                       # (B, T, 2*dim_out)
        x_gated, x_val = x_proj.chunk(2, dim=-1)       # İkiye ayır
        x_swiglu = F.silu(x_gated) * x_val             # SwiGLU aktivasyonu

        return self.linear2(x_swiglu) 


class FlashAttentionBlockBase(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout= dropout

    def forward(self, q, k, v,cu_seqlens=None, max_seqlen=None, causal=True):
        return flash_attn_func(q, k, v, causal=causal, dropout_p=self.dropout)


""" 
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

 """

class AttentionBlock(nn.Module):

    def __init__(self, hidden_size=64, n_heads=16, dropout=0.1, max_seq_len=9048, device="cuda", dtype="bfloat16", **kwargs):
        super().__init__()

        self.config = kwargs.get("config")
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.max_seq_len = max_seq_len  

        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // n_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, device=self.device, dtype=self.dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, device=self.device, dtype=self.dtype)
        self.rms_norm1 = RMSNormBlock(hidden_size, device=self.device, dtype=dtype)

      
        self.attn = FlashAttentionBlockBase(dropout=dropout) if self.config.attn_implementation == "flash_attention_2" else  nn.MultiheadAttention(embed_dim=hidden_size, num_heads=self.n_heads, batch_first=True, dropout=dropout, device=self.device, dtype=self.dtype)
        
        self.rope = RotaryPositionalEmbedding(config=self.config, device=self.device, dtype=dtype)

  
        
    # def build_rope_cache(self,  base=10000):
    #     """
    #     RoPE için cos ve sin tablolarını oluşturur.
        
    #     Args:
    #         seq_len (int): Maksimum sequence uzunluğu
    #         dim (int): Attention dim (head_dim)
    #         device (str): GPU/CPU
    #         base (int): Rotary encoding için baz (default: 10_000)

    #     Returns:
    #         cos: [1, 1, seq_len, dim]
    #         sin: [1, 1, seq_len, dim]
    #     """

    #     half_dim = self.head_dim // 2
    #     inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 1.0, device=self.device) / half_dim))
    #     # [seq_len, half_dim]
    #     positions = torch.arange(self.max_seq_len, device=self.device).type_as(inv_freq).unsqueeze(1)  # [seq_len, 1]
    #     freqs = torch.einsum("i,j->ij", positions.squeeze(), inv_freq)  # [seq_len, half_dim]

    #     # [seq_len, dim]
    #     emb = torch.cat([freqs, freqs], dim=-1)  # çiftlenmiş hali
    #     cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
    #     sin = emb.sin()[None, None, :, :]  # [1, 1, seq_len, dim]

    #     return cos, sin 
  

        
        
    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        
        B, T, C = x.size()
      
        x = self.rms_norm1(x)

        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)  # (B, T, 3, H, D)

        # Eğer geçmişten gelen varsa, k ve v'yi birleştir
        if past_key_value is not None:
            past_k, past_v = past_key_value  # (B, T_past, H, D)
            k = torch.cat([past_k, qkv[:, :, 1]], dim=1)
            v = torch.cat([past_v, qkv[:, :, 2]], dim=1)
        else:
            k = qkv[:, :, 1]
            v = qkv[:, :, 2]

        q = qkv[:, :, 0]
        
        
        #cos, sin = self.build_rope_cache()

        position_ids = torch.arange(0, T, device=self.device).unsqueeze(0).expand(B, -1)  # [2, 2048]

        # ve forward'da:
        q, k = self.rope.forward(q, k, position_ids)

        if self.device == "cpu" and self.config.attn_implementation == "flash_attention_2":
            raise ValueError("❌ FlashAttention cannot run on CPU. Please switch to attn_implementation='eager'.")

        #q = q.to(dtype=self.dtype, device=self.device)
        #k = k.to(dtype=self.dtype, device=self.device)
        v = v.to(dtype=self.dtype, device=self.device)


        if self.config.attn_implementation == "flash_attention_2":

            #total_tokens = attention_mask.sum(dim=1)  # her örnekteki gerçek token sayıları, shape: (B,)
            #cu_seqlens = F.pad(total_tokens.cumsum(0), (1, 0), value=0)  # shape: (B+1,)
            #max_seqlen = total_tokens.max().item()  # batch içindeki en uzun dizi
            
            # x'i "packed" formata getir: (total, hidden_dim)
            #x_flat = x.reshape(B * T, C).contiguous()

            assert q.device.type == "cuda", "❌ q is not on CUDA"
            assert k.device.type == "cuda", "❌ k is not on CUDA"
            assert v.device.type == "cuda", "❌ v is not on CUDA"

            out = self.attn(q,k,v, cu_seqlens=None, max_seqlen=None)

        else:
            
            q = q.reshape(B, T, self.n_heads * self.head_dim)  # [1, 5, 768]
            k = k.reshape(B, T, self.n_heads * self.head_dim)
            v = v.reshape(B, T, self.n_heads * self.head_dim)

            seq_len = k.size(1)

            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).to(torch.bool)

            out, _ =  self.attn( q, k, v, attn_mask=causal_mask, key_padding_mask=attention_mask.bool(), need_weights=False)

        out = out.reshape(B, T, C)

        if use_cache:
            # istersen k,v'leri orijinal `qkv` tensoründen tekrar çıkarabilirsin
            k = qkv.view(B, T, 3, self.n_heads, self.head_dim)[:, :, 1]
            v = qkv.view(B, T, 3, self.n_heads, self.head_dim)[:, :, 2]
            return self.o_proj(out), (k, v)

        return self.o_proj(out), None


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape=64, eps = 1e-05, elementwise_affine=True, bias=True, device="cuda", dtype="bfloat16"):
        super().__init__()

        self.device = device
        self.dtype = getattr(torch, dtype)

        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=self.dtype)

    def forward(self, x ):

        return self.ln(x)
    

class RMSNormBlock(nn.Module):

    def __init__(self, normalized_shape=64, eps = 1e-05, elementwise_affine=True, bias=True, device="cuda", dtype="bfloat16"):
        super().__init__()

        self.device = device
        self.dtype = getattr(torch, dtype)

        self.rmsNorm = nn.RMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine,  device=self.device, dtype=self.dtype)

    def forward(self, x ):

        return self.rmsNorm(x)

class FeedforwardNetwork(nn.Module):

    def __init__(self, hidden_size=64, device="cuda", dtype="bfloat16"):
        super().__init__()

        self.device = device
        self.dtype = getattr(torch, dtype)

        self.ln1 = nn.Linear(hidden_size, 4 * hidden_size, device=self.device, dtype=self.dtype)
        self.swiglu = SwiGLU(4 * hidden_size, hidden_size, dtype=dtype, device=self.device)
        self.ln2 = nn.Linear(4 * hidden_size, hidden_size, device=self.device,dtype=self.dtype)

    def forward(self, x):
        
        x = self.ln1(x)
        x = self.swiglu(x)
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
        self.rms_norm1 = RMSNormBlock(normalized_shape=hidden_size, device=device, dtype=dtype)
        self.rms_norm2 = RMSNormBlock(normalized_shape=hidden_size, device=device, dtype=dtype)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=None):
        # RMSNorm → Attention → Residual
        residual = x

        x = self.rms_norm1(x)
       
        attention_out, new_past = self.attention_block(x=x, attention_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache)

        x = residual + self.dropout(attention_out)

        # RMSNorm → Feedforward → Residual
        residual = x
        x = self.rms_norm2(x)

        x = residual + self.dropout(self.feedforward_network(x))

        return x, new_past if use_cache else None


class TokenEmbedding(nn.Module):

    def __init__(self, token_count=1000, hidden_size=64, dtype="bfloat16", device="cuda"):
        super().__init__()

        self.dtype=getattr(torch, dtype)

        self.embedding_layer = torch.nn.Embedding(token_count, hidden_size, device=device, dtype=self.dtype)  # Embedding layer with len(tokenizer) unique words and embeds
        
    def get_weight(self):

        return self.embedding_layer.weight
    
    def forward(self,x):

        x = self.embedding_layer(x)

        return x

""" class PositionEmbedding(nn.Module):

    def __init__(self, max_seq_len=512*4,  hidden_size=64, dtype="bfloat16", device="cuda"):
        super().__init__()

        self.dtype=getattr(torch, dtype)

        self.position_embedding = torch.nn.Embedding(max_seq_len, hidden_size, device=device, dtype=self.dtype)  # Embedding layer with len(tokenizer) unique words and embeds

    def forward(self,x):

        x = self.position_embedding(x)

        return x """
class EmbeddingWrapper(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.weight = embedding_layer.get_weight()
        self.dtype = embedding_layer.get_weight().dtype

    def forward(self, x):
        return self.embedding_layer(x)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, config, device="cuda", dtype="bfloat16"):
        super().__init__()
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.config = config

        self.head_dim = config.hidden_size // config.n_heads
        self.base = 10000
        self.scaling_factor = config.rope_scaling["factor"] if config.rope_scaling else 1.0

        # Başlangıçta belirli bir boyutta cache oluştur
        self.cached_seq_len = config.max_position_embeddings
        self.cos_cached, self.sin_cached = self.build_rotary_embeddings(self.cached_seq_len, self.head_dim)

    def build_rotary_embeddings(self, seq_len, dim):
        position = torch.arange(0, seq_len, device=self.device, dtype=self.dtype)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=self.device, dtype=self.dtype) / dim))

        if self.scaling_factor != 1.0:
            position = position / self.scaling_factor

        freqs = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        sin = emb.sin()[None, None, :, :]  # [1, 1, seq_len, dim]

        return cos.to(self.device), sin.to(self.device)

    def _rotate_half(self, x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def forward(self, q, k, position_ids):
        seq_len_needed = position_ids.max().item() + 1

        if seq_len_needed > self.cached_seq_len:
            # Gerekli uzunluk cache'den büyükse, yeniden hesapla ve güncelle
            print(f"🔄 [RoPE] Yeni cache oluşturuluyor: {seq_len_needed} token")
            self.cached_seq_len = seq_len_needed
            self.cos_cached, self.sin_cached = self.build_rotary_embeddings(seq_len_needed, self.head_dim)

        # Seçili pozisyonlara göre cos/sin al
        cos = self.cos_cached.squeeze(0).squeeze(0)[position_ids].unsqueeze(2)  # [B, T, 1, D]
        sin = self.sin_cached.squeeze(0).squeeze(0)[position_ids].unsqueeze(2)

        # q, k zaten [B, T, H, D] formatında olacak
        q = q.to(self.device).to(self.dtype)
        k = k.to(self.device).to(self.dtype)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot






class EmbeddingLayer(nn.Module):

    def __init__(self, token_count=1000,  max_seq_len=512*4, hidden_size=64, dtype="bfloat16", device="cuda"):
        super().__init__()
        
        self.device = device
        self.dtype = dtype

        self.token_embedding = TokenEmbedding(token_count, hidden_size, dtype=self.dtype, device=self.device)  # Embedding layer with len(tokenizer) unique words and embeds

        #self.position_embedding = PositionEmbedding(max_seq_len, hidden_size, dtype=self.dtype, device=self.device)

    def forward(self,x):
        #positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        
        #x = self.token_embedding(x) + self.position_embedding(positions)
        
        x = self.token_embedding(x) 

        return x
    


class CrispyLLMConfig(PretrainedConfig):
    
    model_type = "crispy"
    _attn_implementation: Optional[str] = "eager" 
    def __init__(self, vocab_size=1000, max_seq_len=512*4,hidden_size=768, max_position_embeddings=4096, rope_scaling= {"type": "linear", "factor": 2.0}, num_hidden_layers=12, device="cuda", decoder_dropout=0.2,attention_dropout = 0.2,dtype="bfloat16", **kwargs):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
        self.use_flash_attention_2=kwargs.get("use_flash_attention_2")
        
        self.layer_norm_bias=True

        self.decoder_dropout = decoder_dropout
        self.attention_dropout = attention_dropout

        #self.attn_implementation= "flash_attention_2" if self.use_flash_attention_2 else "eager"
        self.attn_implementation= kwargs.get("attn_implementation")
        
        self.gradient_checkpointing = False

        self.dtype = dtype
        self.device = device

        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling

        

    
class CrispyForCausalLM(PreTrainedModel, GenerationMixin):
    
    _supports_flash_attn_2 = True
    config_class = CrispyLLMConfig
    supports_gradient_checkpointing = True

    def __init__(self,  config: CrispyLLMConfig, 
        pad_token_id=3,
        bos_token_id=0,
        eos_token_id=1,
        unk_token_id=2, 
        *args, **kwargs):
        super().__init__(config, 
                        pad_token_id=pad_token_id,
                        bos_token_id=bos_token_id,
                        eos_token_id=eos_token_id,
                        unk_token_id=unk_token_id,
                        )

        self.validate_attention_config(config)

        self.gradient_checkpointing = config.gradient_checkpointing

        self.embedding = EmbeddingLayer( token_count=config.vocab_size, max_seq_len = config.max_seq_len, hidden_size = config.hidden_size, device=config.device, dtype=config.dtype )
        self.decoderBlocks = nn.ModuleList([(DecoderBlock(config=config,n_heads=config.n_heads,hidden_size= config.hidden_size, max_seq_len=config.max_seq_len,  device=config.device, dtype=config.dtype)) for i in range(config.num_hidden_layers)])
        
        self.final_ln = RMSNormBlock( normalized_shape=config.hidden_size, device=config.device, dtype=config.dtype)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, device=config.device, dtype=getattr(torch, config.dtype))

        #self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

    def validate_attention_config(self, config):
        
        if config.attn_implementation == "flash_attention_2" and config.device == "cpu":
            raise ValueError(
                "❌ FlashAttention is not supported on CPU.\n"
                "You are using `attn_implementation='flash_attention_2'` with `device='cpu'`, which is incompatible.\n\n"
                "🛠️ Please either:\n"
                "- Use a CUDA-enabled GPU device, or\n"
                "- Change `attn_implementation` to `'eager'` for CPU compatibility.\n\n"
                "✅ Example fix:\n    config.attn_implementation = 'eager'"
            )
        
    def _init_weights(self):
        # ln1 init
        nn.init.xavier_uniform_(self.lm_head.weight)
        

         # lm_head bias init (sıfırla, eğer varsa)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value
        for block in self.decoderBlocks:
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = value




    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        
        
        from huggingface_hub import snapshot_download

        if os.path.isdir(pretrained_model_name_or_path)==False:
            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path, local_dir=pretrained_model_name_or_path, cache_dir=pretrained_model_name_or_path)

        quantization_config = kwargs.pop("quantization_config", None)

        if quantization_config is not None:
            import bitsandbytes as bnb
            bnb.replace_8bit_linear()
            bnb.replace_with_bnb_linear()

        # Config yükleniyor
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

       

        # Dtype ve diğer özel config ayarları
        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is not None:
            config.torch_dtype = torch_dtype

        # Eğer config içinde attn_implementation ayarı varsa, autoset olarak işaretle
        if hasattr(config, "attn_implementation"):
            config._attn_implementation_autoset = True

        # Model örneği oluşturuluyor
        model = cls(config, *model_args, **kwargs)

        # .safetensors varsa onu kullan, yoksa .bin
        safetensor_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        bin_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")

        if os.path.exists(safetensor_path):
            print("📦 Loading weights from model.safetensors")
            state_dict = safe_load_file(safetensor_path, device="cpu")
        elif os.path.exists(bin_path):
            print("📦 Loading weights from pytorch_model.bin")
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError("Ağırlık dosyası bulunamadı.")

        #print(state_dict)
        
        # Flash / Eager dönüşüm kontrolü
        if any("attn.attn.Wqkv.weight" in k for k in state_dict):
            print("⚡ FlashAttention ağırlıkları tespit edildi.")
            if config.attn_implementation == "eager" or config.attn_implementation == None:
                print("🔁 Flash → Eager çevirisi yapılıyor...")
                state_dict = model.convert_flash_to_eager(state_dict)

        elif any("in_proj_weight" in k for k in state_dict):
            print("🧠 Eager attention ağırlıkları tespit edildi.")
            if config.attn_implementation == "flash_attention_2" or config.attn_implementation == None :
                print("🔁 Eager → Flash çevirisi yapılıyor...")
                state_dict = model.convert_eager_to_flash(state_dict)

        # Ağırlık yükleme
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print(f"⚠️ Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"⚠️ Unexpected keys: {unexpected}")

        return model
    def convert_eager_to_flash(self, state_dict):
            new_state = state_dict.copy()
            for i in range(self.config.num_hidden_layers):
                prefix = f"decoderBlocks.{i}.attention_block.attn"
                # Check if eager weights exist
                if f"{prefix}.in_proj_weight" in state_dict:
                    print(f"🔁 Eager → Flash dönüştürülüyor: blok {i}")
                    W = state_dict[f"{prefix}.in_proj_weight"]
                    b = state_dict[f"{prefix}.in_proj_bias"]
                    # Split W and b into q, k, v
                    q_w, k_w, v_w = W.chunk(3, dim=0)
                    q_b, k_b, v_b = b.chunk(3, dim=0)
                    # Set to flash keys
                    new_state[f"{prefix}.attn.Wqkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
                    new_state[f"{prefix}.attn.Wqkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)
                    new_state[f"{prefix}.attn.out_proj.weight"] = state_dict[f"{prefix}.out_proj.weight"]
                    new_state[f"{prefix}.attn.out_proj.bias"] = state_dict[f"{prefix}.out_proj.bias"]
                    # Remove old
                    del new_state[f"{prefix}.in_proj_weight"]
                    del new_state[f"{prefix}.in_proj_bias"]
                    del new_state[f"{prefix}.out_proj.weight"]
                    del new_state[f"{prefix}.out_proj.bias"]
            return new_state

    def convert_flash_to_eager(self,state_dict):
        new_state = state_dict.copy()
        for i in range(self.config.num_hidden_layers):
            prefix = f"decoderBlocks.{i}.attention_block.attn"
            if f"{prefix}.attn.Wqkv.weight" in state_dict:
                print(f"🔁 Flash → Eager dönüştürülüyor: blok {i}")
                Wqkv = state_dict[f"{prefix}.attn.Wqkv.weight"]
                bqkv = state_dict[f"{prefix}.attn.Wqkv.bias"]
                q_w, k_w, v_w = Wqkv.chunk(3, dim=0)
                q_b, k_b, v_b = bqkv.chunk(3, dim=0)
                new_state[f"{prefix}.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
                new_state[f"{prefix}.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
                new_state[f"{prefix}.out_proj.weight"] = state_dict[f"{prefix}.attn.out_proj.weight"]
                new_state[f"{prefix}.out_proj.bias"] = state_dict[f"{prefix}.attn.out_proj.bias"]
                del new_state[f"{prefix}.attn.Wqkv.weight"]
                del new_state[f"{prefix}.attn.Wqkv.bias"]
                del new_state[f"{prefix}.attn.out_proj.weight"]
                del new_state[f"{prefix}.attn.out_proj.bias"]
        return new_state
    
    def tie_weights(self):
        self.lm_head.weight = self.embedding.token_embedding.get_weight() 

        print("✅ Weights tied")
        
    def get_input_embeddings(self):
        return EmbeddingWrapper(self.embedding.token_embedding)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False, token_type_ids=None, **kwargs):

        x = self.embedding(input_ids)

        """  has_nan = torch.isnan(x).any()
        print("NaN var mı? ➤", has_nan.item())  # True veya False döner

        has_inf = torch.isinf(x).any()
        print("Inf var mı? ➤", has_inf.item()) """

        

        new_past_key_values = []
        for i, block in enumerate(self.decoderBlocks):
            past = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def custom_forward(*inputs):
                    return block(*inputs, use_cache=False)
                x, _ = torch.utils.checkpoint.checkpoint(custom_forward, x, attention_mask, past)
            else:
                x, new_past = block(x, attention_mask=attention_mask, past_key_value=past, use_cache=use_cache)
            
            if use_cache:
                new_past_key_values.append(new_past)

        """ has_nan = torch.isnan(x).any()
        print("NaN var mı? ➤", has_nan.item())  # True veya False döner

        has_inf = torch.isinf(x).any()
        print("Inf var mı? ➤", has_inf.item()) """

        x = self.final_ln(x)

        logits = self.lm_head(x)

        #logits = logits.clamp(min=-10.0, max=10.0)

        #logits = self.softmax(logits)
        # print("Max logits:", logits.max())
        # print("Min logits:", logits.min())
        # print("Any NaN:", torch.isnan(logits).any())
        
        
        assert not torch.isnan(logits).any(), "NaN var"
        assert not torch.isinf(logits).any(), "Inf var"

        
        loss= None
        if labels is not None:
            
            if torch.any(labels >= self.config.vocab_size):
                print("🚨 Hatalı label bulundu! Maksimum label:", labels.max().item())
                print("🔢 Vocab size:", self.config.vocab_size)
                raise ValueError("Label değeri vocab_size'dan büyük!")


            labels = labels.clone()

            # SHIFT: logits ve labels kaydırılıyor
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            #labels[labels == self.config.pad_token_id] = -100
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None
        )
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None):
        old_embeddings = self.embedding.token_embedding.embedding_layer
        old_num_tokens, embedding_dim = old_embeddings.weight.size()

        # Yeni embedding katmanını oluştur
        new_embeddings = nn.Embedding(new_num_tokens, embedding_dim,
                                      device=old_embeddings.weight.device,
                                      dtype=old_embeddings.weight.dtype)

        # Ortak kısmı kopyala
        num_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_to_copy] = old_embeddings.weight.data[:num_to_copy]

        # Modele yeni embedding'i ata
        self.embedding.token_embedding.embedding_layer = new_embeddings

        # lm_head'i de eşle (eğer paylaşılmışsa)
        self.lm_head = nn.Linear(embedding_dim, new_num_tokens,
                                 device=new_embeddings.weight.device,
                                 dtype=new_embeddings.weight.dtype)
        
        self.tie_weights()
        return self.get_input_embeddings()

    

if __name__ == '__main__':

    from transformers import AutoConfig, AutoModelForCausalLM
 
    from transformers import PreTrainedTokenizerFast
    from transformers import XLMRobertaTokenizer

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    AutoConfig.register("crispy", CrispyLLMConfig)
    AutoModelForCausalLM.register(CrispyLLMConfig, CrispyForCausalLM)


    crispy_config = CrispyLLMConfig(attn_implementation="flash_attention_2", use_flash_attention_2=True, vocab_size=len(tokenizer.get_vocab()), n_heads=16, max_seq_len=1024, hidden_size=64*16, num_hidden_layers=16, dtype="bfloat16")

    crispy_config._attn_implementation_autoset = True  # 👈 Buraya ekliyorsun

    model = AutoModelForCausalLM.from_config(crispy_config)


    inputs = tokenizer("Selam nasılsın", max_length=1024, padding="max_length",return_tensors="pt")



    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print(inputs)

    print(model.forward(**inputs))

    param = model.get_input_embeddings()
    embedding_weight = param if isinstance(param, nn.Parameter) else param.weight

    print(model.tie_weights())
    print(embedding_weight)

    # print(model)

    # # Prompt
    # prompt = "Türkiye'nin başkenti"
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # # Hugging Face generate
    # output = model.generate(
    #     **inputs,
    #     max_new_tokens=512,
    #     do_sample=False,  # greedy decoding
    #     use_cache=True,   # past_key_values kullanımı
    #     eos_token_id=tokenizer.eos_token_id,
    # )

    # # Sonucu çöz
    # print(tokenizer.decode(output[0], skip_special_tokens=True))