import torch
from torch import nn

from .blocks import ConvBlock, ResidualBlock


class FrequencyAwareResidualRefinement(nn.Module):
    def __init__(self, feature_channels, hidden_channels=64, residual_scale=0.25):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            ConvBlock(3 + feature_channels + 1, hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 3, 3, padding=1),
        )

    def forward(self, coarse, shallow_feature, valid_mask):
        residual = torch.tanh(self.net(torch.cat([coarse, shallow_feature, valid_mask], dim=1)))
        hole = 1.0 - valid_mask
        refined = torch.clamp(coarse + hole * self.residual_scale * residual, -1.0, 1.0)
        return refined, residual

