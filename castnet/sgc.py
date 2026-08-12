import math

import torch
from torch import nn


class SparseGlobalContext(nn.Module):
    """All-token queries attend to a learned top-k context subset."""

    def __init__(self, channels, heads=4, retain_ratio=0.10):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        if not 0 < retain_ratio <= 1:
            raise ValueError("retain_ratio must be in (0, 1]")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.retain_ratio = retain_ratio
        self.norm = nn.LayerNorm(channels)
        self.scorer = nn.Linear(channels, 1)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, feature, return_diagnostics=False):
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        z = self.norm(tokens)
        scores = self.scorer(z).squeeze(-1)
        count = max(1, int(math.ceil(tokens.shape[1] * self.retain_ratio)))
        indices = scores.topk(count, dim=1, sorted=True).indices
        selected = z.gather(1, indices.unsqueeze(-1).expand(-1, -1, c))
        selected_scores = torch.sigmoid(scores.gather(1, indices)).unsqueeze(-1)
        selected = selected * (1.0 + selected_scores - selected_scores.detach())

        q = self.q(z).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(selected).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(selected).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        attention = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        contextual = torch.matmul(attention, v).transpose(1, 2).reshape(b, h * w, c)
        output = self.out(contextual).transpose(1, 2).reshape(b, c, h, w) + feature
        if return_diagnostics:
            return output, {"indices": indices, "scores": scores, "attention": attention}
        return output

