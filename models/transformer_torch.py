import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(x.shape[-1])

        weights = torch.softmax(
            scores,
            dim=-1
        )

        return torch.matmul(weights, v)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.attention = Attention(d_model)

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.ffn2 = nn.Linear(d_ff, d_model)

        self.norm2 = nn.LayerNorm(d_model)

        # Deliberate Identity for Graph Normalization.
        self.dropout_placeholder = nn.Identity()

    def forward(self, x):
        residual = x

        attention_output = self.attention(x)

        x = residual + attention_output
        x = self.norm1(x)

        residual = x

        x = self.ffn1(x)
        x = self.gelu(x)
        x = self.ffn2(x)

        x = residual + x
        x = self.norm2(x)

        x = self.dropout_placeholder(x)

        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size=1000,
        max_seq_len=32,
        d_model=64,
        d_ff=256,
        num_blocks=2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Parameter(
            torch.randn(max_seq_len, d_model)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff)
            for _ in range(num_blocks)
        ])

        self.output = nn.Linear(
            d_model,
            vocab_size
        )

        # Deliberately dead computation for DCE.
        self.debug = nn.Linear(
            d_model,
            d_model
        )

    def forward(self, token_ids):

        x = self.embedding(token_ids)

        x = x + self.position_embedding[:x.shape[1]]

        for block in self.blocks:
            x = block(x)

        debug_value = self.debug(x)

        logits = self.output(x)

        return logits