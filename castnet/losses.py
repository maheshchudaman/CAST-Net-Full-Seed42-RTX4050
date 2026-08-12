from functools import lru_cache

import torch
from torch.nn import functional as F


def hole_l1(prediction, target, valid_mask, eps=1e-6):
    hole = 1.0 - valid_mask
    numerator = (prediction - target).abs().mul(hole).sum()
    denominator = hole.sum() * prediction.shape[1] + eps
    return numerator / denominator


def sobel_edges(image):
    gray = image.mean(dim=1, keepdim=True)
    kx = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-6)


def edge_charbonnier(prediction, target, valid_mask, eps=1e-3):
    hole = 1.0 - valid_mask
    error = sobel_edges(prediction) - sobel_edges(target)
    return (torch.sqrt(error.square() + eps * eps) * hole).sum() / (hole.sum() + 1e-6)


@lru_cache(maxsize=16)
def _cpu_dct_matrix(size):
    n = torch.arange(size, dtype=torch.float32)
    k = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    matrix = torch.cos(torch.pi / size * (n + 0.5) * k)
    matrix[0] *= (1.0 / size) ** 0.5
    matrix[1:] *= (2.0 / size) ** 0.5
    return matrix


def dct2(image):
    h, w = image.shape[-2:]
    dh = _cpu_dct_matrix(h).to(device=image.device, dtype=image.dtype)
    dw = _cpu_dct_matrix(w).to(device=image.device, dtype=image.dtype)
    return torch.matmul(torch.matmul(dh, image), dw.transpose(0, 1))


def frequency_l1(prediction, target, high_frequency_start=0.35):
    pred = dct2(prediction)
    truth = dct2(target)
    h, w = prediction.shape[-2:]
    yy = torch.arange(h, device=prediction.device).view(h, 1) / max(h - 1, 1)
    xx = torch.arange(w, device=prediction.device).view(1, w) / max(w - 1, 1)
    weights = ((yy + xx) / 2 >= high_frequency_start).to(prediction.dtype)
    return ((pred - truth).abs() * weights).sum() / (weights.sum() * prediction.shape[0] * prediction.shape[1] + 1e-6)


def generator_objective(output, target, valid_mask, lambda_frequency=0.1, lambda_edge=0.1):
    reconstruction = hole_l1(output["completed"], target, valid_mask)
    frequency = frequency_l1(output["completed"], target)
    edge = edge_charbonnier(output["completed"], target, valid_mask)
    total = reconstruction + lambda_frequency * frequency + lambda_edge * edge
    return {"total": total, "reconstruction": reconstruction, "frequency": frequency, "edge": edge}
