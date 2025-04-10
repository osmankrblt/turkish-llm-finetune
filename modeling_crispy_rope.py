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
        
        self.rope = RotaryPositionalEmbedding(device=self.device, dtype=dtype)

  

    def build_rope_cache(self,  base=10000):
        """
        RoPE için cos ve sin tablolarını oluşturur.
        
        Args:
            seq_len (int): Maksimum sequence uzunluğu
            dim (int): Attention dim (head_dim)
            device (str): GPU/CPU
            base (int): Rotary encoding için baz (default: 10_000)

        Returns:
            cos: [1, 1, seq_len, dim]
            sin: [1, 1, seq_len, dim]
        """

        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 1.0, device=self.device) / half_dim))
        # [seq_len, half_dim]
        positions = torch.arange(self.max_seq_len, device=self.device).type_as(inv_freq).unsqueeze(1)  # [seq_len, 1]
        freqs = torch.einsum("i,j->ij", positions.squeeze(), inv_freq)  # [seq_len, half_dim]

        # [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)  # çiftlenmiş hali
        cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        sin = emb.sin()[None, None, :, :]  # [1, 1, seq_len, dim]

        return cos, sin
  

        
        
    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):

        B, T, C = x.size()
        print(B,T,C)
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
        
        
        cos, sin = self.build_rope_cache()

        position_ids = torch.arange(0, T, device=self.device).unsqueeze(0).expand(B, -1)  # [2, 2048]

        # ve forward'da:
        q, k = self.rope.forward(q, k, cos, sin, position_ids)
       
        if self.config.attn_implementation == "flash_attention_2":

            total_tokens = attention_mask.sum(dim=1)  # her örnekteki gerçek token sayıları, shape: (B,)
            cu_seqlens = F.pad(total_tokens.cumsum(0), (1, 0), value=0)  # shape: (B+1,)
            max_seqlen = total_tokens.max().item()  # batch içindeki en uzun dizi
            
            # x'i "packed" formata getir: (total, hidden_dim)
            x_flat = x.reshape(B * T, C).contiguous()
          
            out = self.attn(q,k,v, cu_seqlens=cu_seqlens.to(torch.int32), max_seqlen=max_seqlen)

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

    def forward(self, x, attention_mask=None,past_key_value=None, use_cache=None):
        # RMSNorm → Attention → Residual
        residual = x

        x = self.rms_norm1(x)

        attention_out, new_past = self.attention_block(x, attention_mask, past_key_value=past_key_value,
        use_cache=use_cache)

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
    
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, device="cuda", dtype="bfloat16"):
        super().__init__()
        self.device = device
        self.dtype = getattr(torch, dtype)

    def _rotate_half(self, tensor):
        """
        Tensor'un son boyutunu ikiye bölerek yarısını rotasyon uygular:
        [x1, x2] → [-x2, x1]
        """
        first_half = tensor[..., :tensor.shape[-1] // 2]
        second_half = tensor[..., tensor.shape[-1] // 2:]
        return torch.cat((-second_half, first_half), dim=-1)

    def forward(self, q, k, cos, sin, position_ids):
        """
        Rotary Positional Embedding uygular.

        Args:
            q: Query tensor [batch, seq_len, heads, dim]
            k: Key tensor   [batch, seq_len, heads, dim]
            cos: [1, 1, seq_len, dim] formatında önceden hesaplanmış cos değerleri
            sin: [1, 1, seq_len, dim] formatında önceden hesaplanmış sin değerleri
            position_ids: [batch, seq_len] pozisyon index tensorü

        Returns:
            Tuple: (q_rotated, k_rotated)
        """
        # [seq_len, dim]
        cos = cos.squeeze(0).squeeze(0).to(self.device).to(self.dtype)
        sin = sin.squeeze(0).squeeze(0).to(self.device).to(self.dtype)

        # [batch, seq_len, dim]
        cos = cos[position_ids].unsqueeze(2).to(self.device).to(self.dtype)  # [B, T, 1, D]
        sin = sin[position_ids].unsqueeze(2).to(self.device).to(self.dtype)

        # q, k zaten [B, T, H, D] formatında olacak
        q = q.to(self.device).to(self.dtype)
        k = k.to(self.device).to(self.dtype)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated




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
    def __init__(self, vocab_size=1000, max_seq_len=512*4,hidden_size=768, num_hidden_layers=12, device="cuda", dtype="bfloat16", **kwargs):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        

        self.use_flash_attention_2=kwargs.get("use_flash_attention_2")
        
        #self.attn_implementation= "flash_attention_2" if self.use_flash_attention_2 else "eager"
        self.attn_implementation= kwargs.get("attn_implementation")
        
        

        self.dtype = dtype
        self.device = device

class CrispyForCausalLM(PreTrainedModel, GenerationMixin):
    
    _supports_flash_attn_2 = True
    config_class = CrispyLLMConfig
    
    def __init__(self,  config: CrispyLLMConfig, *args, **kwargs):
        super().__init__(config)

        self.embedding = EmbeddingLayer( token_count=config.vocab_size, max_seq_len = config.max_seq_len, hidden_size = config.hidden_size, device=config.device, dtype=config.dtype )
        self.decoderBlocks = nn.ModuleList([ (DecoderBlock(config=config,n_heads=config.n_heads,hidden_size= config.hidden_size, max_seq_len=config.max_seq_len,  device=config.device, dtype=config.dtype)) for i in range(config.num_hidden_layers)])
        
        self.final_ln = RMSNormBlock(normalized_shape=config.hidden_size, device=config.device, dtype=config.dtype)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, device=config.device, dtype=getattr(torch, config.dtype))

        self.softmax = nn.Softmax(dim=-1)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):

        from transformers.modeling_utils import load_state_dict
        from safetensors.torch import load_file as safe_load_file
        import os


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

        # Flash / Eager dönüşüm kontrolü
        if any("attn.attn.Wqkv.weight" in k for k in state_dict):
            print("⚡ FlashAttention ağırlıkları tespit edildi.")
            if config.attn_implementation == "eager":
                print("🔁 Flash → Eager çevirisi yapılıyor...")
                state_dict = model.convert_flash_to_eager(state_dict)

        elif any("in_proj_weight" in k for k in state_dict):
            print("🧠 Eager attention ağırlıkları tespit edildi.")
            if config.attn_implementation == "flash_attention_2":
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
        
    def get_input_embeddings(self):
        return self.embedding.token_embedding.get_weight()  # veya input_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False, token_type_ids=None, **kwargs):

        x = self.embedding(input_ids)
      
        new_past_key_values = []
        for i, block in enumerate(self.decoderBlocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, new_past = block(x, attention_mask=attention_mask, past_key_value=past, use_cache=use_cache)
            if use_cache:
                new_past_key_values.append(new_past)

        x = self.final_ln(x)

        logits = self.lm_head(x)

        logits = self.softmax(logits)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None
        )
    

if __name__ == '__main__':

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained("MyLLM/CrispyTokenizer")
    crispy_config = CrispyLLMConfig(attn_implementation="flash_attention_2", vocab_size=len(tokenizer.get_vocab()), n_heads=8, max_seq_len=1024, hidden_size=64*16, num_hidden_layers=8, dtype="bfloat16", device="cuda")

    #crispy_config._attn_implementation_autoset = True  # 👈 Buraya ekliyorsun
    model = CrispyForCausalLM(crispy_config)

    inputs = tokenizer("Selam nasılsın", max_length=1024, padding="max_length",return_tensors="pt")

    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print(inputs)

    print(model.forward(**inputs))

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