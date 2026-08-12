from dataclasses import dataclass

import torch
from torch import nn

from .blocks import ConvBlock, ResidualBlock, UpBlock
from .csam import CrossScaleAffinityMixer
from .farr import FrequencyAwareResidualRefinement
from .sgc import SparseGlobalContext


@dataclass
class CASTNetConfig:
    base_channels: int = 64
    attention_heads: int = 4
    retain_ratio: float = 0.10
    csam_temperature: float = 0.07
    residual_scale: float = 0.25
    use_sgc: bool = True
    use_csam: bool = True
    use_farr: bool = True


class CASTNet(nn.Module):
    """Context-Aware Sparse-Transformer network for image inpainting.

    valid_mask uses 1 for observed pixels and 0 for the hole.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or CASTNetConfig()
        b = config.base_channels
        self.config = config
        self.enc1 = nn.Sequential(ConvBlock(4, b, 5, norm=False), ResidualBlock(b))
        self.enc2 = nn.Sequential(ConvBlock(b, b * 2, 4, 2), ResidualBlock(b * 2))
        self.enc3 = nn.Sequential(ConvBlock(b * 2, b * 4, 4, 2), ResidualBlock(b * 4))
        self.sgc = SparseGlobalContext(b * 4, config.attention_heads, config.retain_ratio) if config.use_sgc else None
        self.csam = CrossScaleAffinityMixer(b * 2, b * 4, config.csam_temperature) if config.use_csam else None
        self.fusion = nn.Sequential(ConvBlock(b * 8, b * 4, 1), ResidualBlock(b * 4))
        self.up2 = UpBlock(b * 4, b * 2, b * 2)
        self.up1 = UpBlock(b * 2, b, b)
        self.coarse_head = nn.Sequential(ConvBlock(b, b // 2), nn.Conv2d(b // 2, 3, 3, padding=1), nn.Tanh())
        self.farr = FrequencyAwareResidualRefinement(b, b, config.residual_scale) if config.use_farr else None

    def forward(self, corrupted, valid_mask, return_diagnostics=False):
        if corrupted.ndim != 4 or valid_mask.ndim != 4:
            raise ValueError("corrupted and valid_mask must be BCHW tensors")
        if valid_mask.shape[1] != 1 or corrupted.shape[1] != 3:
            raise ValueError("expected RGB image and one-channel validity mask")
        x = torch.cat([corrupted, valid_mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        if self.sgc is None:
            global_feature = e3
            sgc_diag = {"enabled": False}
        elif return_diagnostics:
            global_feature, sgc_diag = self.sgc(e3, True)
            sgc_diag["enabled"] = True
        else:
            global_feature = self.sgc(e3)
        if self.csam is None:
            local_feature = e3
            csam_diag = {"enabled": False}
        elif return_diagnostics:
            local_feature, csam_diag = self.csam(e3, e2, valid_mask, True)
            csam_diag["enabled"] = True
        else:
            local_feature = self.csam(e3, e2, valid_mask)
        fused = self.fusion(torch.cat([global_feature, local_feature], dim=1))
        d2 = self.up2(fused, e2)
        d1 = self.up1(d2, e1)
        coarse_prediction = self.coarse_head(d1)
        coarse = valid_mask * corrupted + (1.0 - valid_mask) * coarse_prediction
        if self.farr is None:
            refined = coarse
            residual = torch.zeros_like(coarse)
        else:
            refined, residual = self.farr(coarse, d1, valid_mask)
        completed = valid_mask * corrupted + (1.0 - valid_mask) * refined
        output = {"prediction": refined, "coarse": coarse, "completed": completed, "residual": residual}
        if return_diagnostics:
            output["diagnostics"] = {
                "sgc": sgc_diag,
                "csam": csam_diag,
                "farr": {"enabled": self.farr is not None},
            }
        return output

