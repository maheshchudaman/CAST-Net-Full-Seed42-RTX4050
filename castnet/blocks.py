import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if norm:
            layers.append(nn.GroupNorm(min(8, out_channels), out_channels))
        layers.append(nn.SiLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(channels, channels), nn.Conv2d(channels, channels, 3, padding=1))
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        return F.silu(self.norm(x + self.net(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBlock(in_channels + skip_channels, out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))

