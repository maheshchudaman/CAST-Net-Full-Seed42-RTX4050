import torch

from .losses import sobel_edges


def masked_psnr(prediction, target, valid_mask, eps=1e-8):
    hole = 1.0 - valid_mask
    mse = ((prediction - target).square() * hole).sum() / (hole.sum() * prediction.shape[1] + eps)
    return 10.0 * torch.log10(4.0 / (mse + eps))


def standard_ssim(prediction, target, reduction="elementwise_mean"):
    """Windowed SSIM from TorchMetrics, with inputs converted to [0, 1]."""
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
    except ImportError as exc:
        raise RuntimeError("standard SSIM requires torchmetrics; install requirements.txt") from exc
    prediction = ((prediction + 1.0) / 2.0).clamp(0, 1)
    target = ((target + 1.0) / 2.0).clamp(0, 1)
    return structural_similarity_index_measure(
        prediction,
        target,
        data_range=1.0,
        gaussian_kernel=True,
        sigma=1.5,
        kernel_size=11,
        reduction=reduction,
    )


def masked_edge_mae(prediction, target, valid_mask, eps=1e-8):
    hole = 1.0 - valid_mask
    difference = (sobel_edges(prediction) - sobel_edges(target)).abs()
    numerator = (difference * hole).sum(dim=(1, 2, 3))
    denominator = hole.sum(dim=(1, 2, 3)).clamp_min(eps)
    return numerator / denominator

