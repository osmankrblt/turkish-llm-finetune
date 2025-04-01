import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):

    def __init__(self, token_count=1000, embed_size=64):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(token_count, embed_size)  # Embedding layer with len(tokenizer) unique words and embeds

    def forward(self,x):

        x = self.embedding_layer(x)

        return x

class PositionEmbedding(nn.Module):

    def __init__(self, max_seq_len=512*4,  embed_size=64):
        super().__init__()
        self.position_embedding = torch.nn.Embedding(max_seq_len, embed_size)  # Embedding layer with len(tokenizer) unique words and embeds

    def forward(self,x):

        x = self.position_embedding(x)

        return x


class EmbeddingLayer(nn.Module):

    def __init__(self, token_count=1000,  max_seq_len=512*4, embed_size=64):
        super().__init__()
        self.token_embedding = TokenEmbedding(token_count, embed_size)  # Embedding layer with len(tokenizer) unique words and embeds

        self.position_embedding = PositionEmbedding(max_seq_len, embed_size)

    def forward(self,x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)

        return x

if __name__=="__main__":

    # Create an instance of EmbeddingLayer

    embedLayer = EmbeddingLayer(1000,64)

    torchTensor = torch.tensor([[15,20,54]],  dtype=torch.long)
    
    print(embedLayer.forward(torchTensor))