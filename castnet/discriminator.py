import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        layers = []
        channels = [4, base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        for index in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[index], channels[index + 1], 4, 2 if index < 3 else 1, 1))
            if index:
                layers.append(nn.GroupNorm(min(8, channels[index + 1]), channels[index + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(channels[-1], 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, image, valid_mask):
        return self.net(torch.cat([image, valid_mask], dim=1))


def discriminator_hinge(real_logits, fake_logits):
    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def generator_hinge(fake_logits):
    return -fake_logits.mean()
