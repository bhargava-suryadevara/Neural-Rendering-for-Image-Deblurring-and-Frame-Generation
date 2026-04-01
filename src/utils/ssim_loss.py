"""
Differentiable SSIM for training (PyTorch). Not used for evaluation; see metrics.py for skimage SSIM.
"""

import torch
import torch.nn.functional as F

_SSIM_C1 = 0.01**2
_SSIM_C2 = 0.03**2


def _gaussian_window_2d(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d


def differentiable_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Mean SSIM over batch, channels, and spatial dims. Fully differentiable.

    Args:
        pred: (B, C, H, W), same shape as target, typically in [0, 1].
        target: (B, C, H, W).

    Returns:
        Scalar tensor: mean SSIM in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    _, c, _, _ = pred.shape
    device = pred.device
    dtype = pred.dtype

    window = _gaussian_window_2d(window_size, sigma, device, dtype)
    window = window.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_x = F.conv2d(pred, window, padding=pad, groups=c)
    mu_y = F.conv2d(target, window, padding=pad, groups=c)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=pad, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_xy

    ssim_map = ((2 * mu_xy + _SSIM_C1) * (2 * sigma_xy + _SSIM_C2)) / (
        (mu_x_sq + mu_y_sq + _SSIM_C1) * (sigma_x_sq + sigma_y_sq + _SSIM_C2) + 1e-8
    )
    return ssim_map.mean()
