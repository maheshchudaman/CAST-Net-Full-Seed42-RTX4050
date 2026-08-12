import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossScaleAffinityMixer(nn.Module):
    """Mask-aware texture retrieval from observed source locations."""

    def __init__(self, fine_channels, channels, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.fine_project = nn.Conv2d(fine_channels, channels, 1)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, bottleneck, fine_feature, valid_mask, return_diagnostics=False):
        fine = self.fine_project(fine_feature)
        fine = F.interpolate(fine, bottleneck.shape[-2:], mode="bilinear", align_corners=False)
        mixed = bottleneck + fine
        b, c, h, w = mixed.shape
        valid = F.interpolate(valid_mask.float(), (h, w), mode="nearest").flatten(1).bool()
        has_valid = valid.any(dim=1, keepdim=True)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~has_valid.squeeze(1)

        q = F.normalize(self.query(bottleneck).flatten(2).transpose(1, 2), dim=-1)
        k = F.normalize(self.key(mixed).flatten(2).transpose(1, 2), dim=-1)
        v = self.value(mixed).flatten(2).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(1, 2)) / max(self.temperature, 1e-6)
        logits = logits.masked_fill(~safe_valid[:, None, :], -1e4)
        weights = torch.softmax(logits, dim=-1)
        transported = torch.matmul(weights, v).transpose(1, 2).reshape(b, c, h, w)
        transported = transported * has_valid.view(b, 1, 1, 1)
        hole = 1.0 - F.interpolate(valid_mask.float(), (h, w), mode="nearest")
        gate = self.gate(torch.cat([bottleneck, transported], dim=1))
        output = bottleneck + hole * gate * self.output(transported)
        if return_diagnostics:
            return output, {"affinity": weights, "valid_sources": valid}
        return output

