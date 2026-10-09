import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=4):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(input_dim, hidden_dim)
        )

        self.bias = nn.Parameter(
            torch.randn(hidden_dim)
        )

        self.output_weight = nn.Parameter(
            torch.randn(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x @ self.weight
        x = x + self.bias
        x = torch.relu(x)

        x = x @ self.output_weight

        return x