import math

import torch
import torch.nn as nn


def expand_time(t, B):
    if isinstance(t, float):
        t = t * torch.ones((B,))
    elif torch.is_tensor(t):
        if not t.shape:
            t = t * torch.ones((B,))
        else:
            assert t.shape[0] == B
            t = t.reshape(B)
    return t


class NoiseLevelEncoding(torch.nn.Module):
    """
    From: https://github.com/microsoft/mattergen/blob/ac9ddd406171138c3f037d06b9b53fedbbb1c536/mattergen/diffusion/model_utils.py#L54
    # Copyright (c) Microsoft Corporation.
    # Licensed under the MIT License.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
        """
        x = torch.zeros((t.shape[0], self.d_model), device=self.div_term.device)
        x[:, 0::2] = torch.sin(t[:, None] * self.div_term[None])
        x[:, 1::2] = torch.cos(t[:, None] * self.div_term[None])
        return self.dropout(x)


class WeightsMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_embedder = NoiseLevelEncoding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, 1),
        )

    def forward(self, s, t):
        x = torch.cat([self.time_embedder(s), self.time_embedder(t)], dim=-1)
        return self.mlp(x).squeeze(-1)
