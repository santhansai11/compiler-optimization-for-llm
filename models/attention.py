import math
import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()

        self.wq = nn.Parameter(
            torch.randn(d_model, d_model)
        )

        self.wk = nn.Parameter(
            torch.randn(d_model, d_model)
        )

        self.wv = nn.Parameter(
            torch.randn(d_model, d_model)
        )

    def forward(self, x):
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv

        kt = k.transpose(-2, -1)

        scores = q @ kt

        scores = scores / math.sqrt(x.shape[-1])

        weights = torch.softmax(
            scores,
            dim=-1
        )

        output = weights @ v

        return output


class TransformerAttention(nn.Module):

    def __init__(self, d_model=64):
        super().__init__()

        self.d_model = d_model

        self.Wq = nn.Parameter(
            torch.randn(d_model, d_model)
        )

        self.Wk = nn.Parameter(
            torch.randn(d_model, d_model)
        )

        self.Wv = nn.Parameter(
            torch.randn(d_model, d_model)
        )

    def forward(self, x):

        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.d_model)

        weights = torch.softmax(
            scores,
            dim=-1
        )

        output = weights @ V

        return output