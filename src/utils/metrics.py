from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert a CHW tensor in [0,1] to HWC numpy array in [0,1]."""
    img = img.detach().cpu().clamp(0.0, 1.0)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img[0]
    return img.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Slow (scikit-image) — accurate, CPU-only. Use only when you need exact
# per-image values for display labels (a handful of images).
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR for a single CHW image pair."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """scikit-image SSIM for a single CHW image pair. Accurate but slow (CPU)."""
    pred_np   = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)
    return float(
        structural_similarity(target_np, pred_np, channel_axis=2, data_range=1.0)
    )


def batch_psnr_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float]:
    """
    Average PSNR/SSIM over a batch using scikit-image.
    Kept for backward-compatibility and for labelling saved images.
    For the main evaluation loop use fast_batch_psnr_ssim instead.
    """
    assert preds.shape == targets.shape
    b = preds.shape[0]
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    for i in range(b):
        psnr_vals.append(psnr(preds[i], targets[i]))
        ssim_vals.append(ssim(preds[i], targets[i]))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def per_image_psnr_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[List[float], List[float]]:
    """
    Per-image PSNR and SSIM using scikit-image. Returns two lists of length B.
    Use only for the small number of images you want to annotate with labels.
    """
    assert preds.shape == targets.shape
    b = preds.shape[0]
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    for i in range(b):
        psnr_vals.append(psnr(preds[i], targets[i]))
        ssim_vals.append(ssim(preds[i], targets[i]))
    return psnr_vals, ssim_vals


# ---------------------------------------------------------------------------
# Fast (PyTorch) — runs on MPS / CUDA / CPU. Use this in the main eval loop.
# ---------------------------------------------------------------------------

_SSIM_C1 = 0.01 ** 2
_SSIM_C2 = 0.03 ** 2


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = g.unsqueeze(1) @ g.unsqueeze(0)          # (win, win)
    return w2d.expand(channels, 1, window_size, window_size).contiguous()


def fast_psnr_batch(preds: torch.Tensor, targets: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Batch-level PSNR computed fully on-device (MPS/CUDA/CPU). ~100× faster than
    the scikit-image path for large batches.
    """
    mse = F.mse_loss(preds, targets)
    if mse.item() == 0:
        return 100.0
    return (20.0 * torch.log10(torch.tensor(max_val, device=preds.device))
            - 10.0 * torch.log10(mse)).item()


def fast_ssim_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Mean SSIM over a batch, computed fully on-device. Matches the differentiable
    SSIM used during training. Much faster than scikit-image on MPS/CUDA.
    """
    if preds.shape != targets.shape:
        raise ValueError("preds and targets must have the same shape")
    _, c, _, _ = preds.shape
    device, dtype = preds.device, preds.dtype
    window = _gaussian_window(window_size, sigma, device, dtype, c)
    pad = window_size // 2

    mu_x  = F.conv2d(preds,   window, padding=pad, groups=c)
    mu_y  = F.conv2d(targets, window, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sx2 = F.conv2d(preds   * preds,   window, padding=pad, groups=c) - mu_x2
    sy2 = F.conv2d(targets * targets, window, padding=pad, groups=c) - mu_y2
    sxy = F.conv2d(preds   * targets, window, padding=pad, groups=c) - mu_xy

    ssim_map = ((2 * mu_xy + _SSIM_C1) * (2 * sxy + _SSIM_C2)) / (
        (mu_x2 + mu_y2 + _SSIM_C1) * (sx2 + sy2 + _SSIM_C2) + 1e-8
    )
    return float(ssim_map.mean().item())


def fast_batch_psnr_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float]:
    """
    Fast on-device PSNR + SSIM for the main evaluation loop.
    Stays on MPS/CUDA — no CPU transfer or scikit-image overhead.
    """
    return fast_psnr_batch(preds, targets), fast_ssim_batch(preds, targets)
