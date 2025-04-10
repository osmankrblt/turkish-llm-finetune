from typing import Optional, OrderedDict
import torch
import torch.nn as nn
import torch.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import sys

sys.path.append("MyLLM/CrispyModeling/Embedding")
sys.path.append("MyLLM/CrispyModeling/DecoderBlock")
sys.path.append("./CrispyModeling/Embedding")
sys.path.append("./CrispyModeling/DecoderBlock")

from embedding import EmbeddingLayer    
from decoder import DecoderBlock, LayerNorm

from transformers.modeling_outputs import CausalLMOutputWithPast

class CrispyLLMConfig(PretrainedConfig):
    
    model_type = "crispy"
    _attn_implementation: Optional[str] = "eager" 
    def __init__(self, vocab_size=1000, max_seq_len=512*4,hidden_size=768, num_hidden_layers=12, device="cuda", dtype="bfloat16", **kwargs):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dtype = dtype
        self.device = device

        self.use_flash_attention=False
        self.attn_implementation=kwargs.get("attn_implementation")
        
       

class CrispyForCausalLM(PreTrainedModel):
    
    _supports_flash_attn_2 = True
    config_class = CrispyLLMConfig
    
    def __init__(self, config):
        super().__init__(config)

        self.embedding = EmbeddingLayer(token_count=config.vocab_size, max_seq_len = config.max_seq_len, hidden_size = config.hidden_size, device=config.device, dtype=config.dtype )
        self.decoderBlocks = nn.ModuleList([ (DecoderBlock(config=config,n_heads=config.n_heads,hidden_size= config.hidden_size, max_seq_len=config.max_seq_len,  device=config.device, dtype=config.dtype)) for i in range(config.num_hidden_layers)])
        
        self.final_ln = LayerNorm(normalized_shape=config.hidden_size, device=config.device, dtype=config.dtype)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, device=config.device, dtype=getattr(torch, config.dtype))

        self.softmax = nn.Softmax(dim=-1)

    def tie_weights(self):
        #self.lm_head.weight = self.embedding.token_embedding.get_weight().weight
        pass

    def get_input_embeddings(self):
        return self.embedding.token_embedding.get_weight()  # veya input_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None):

        x = self.embedding(input_ids)
      
        for block in self.decoderBlocks:
            x = block(x, attention_mask=attention_mask.bool())
   
        x = self.final_ln(x)

        x = self.lm_head(x)

        logits = self.softmax(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )
    

    
if __name__ == '__main__':

    from transformers import AutoConfig, AutoModelForCausalLM

    crispy_config = CrispyLLMConfig()
    model = CrispyModel(crispy_config)

    device="cuda"

    input_ids = torch.tensor([[10, 20, 60, 45, 20]], dtype = torch.long, device=device)
    vocab_size = input_ids.max().item() + 1  # 60 + 1 = 61


    print(model.forward(input_ids))



    

    


    
