from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """
    Convert a CHW tensor in [0,1] to HWC numpy array in [0,1].
    """
    img = img.detach().cpu().clamp(0.0, 1.0)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img[0]
    return img.permute(1, 2, 0).numpy()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute PSNR for a single image pair (CHW tensors).
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute SSIM for a single image pair (CHW tensors).
    """
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)
    return float(
        structural_similarity(
            target_np,
            pred_np,
            channel_axis=2,
            data_range=1.0,
        )
    )


def batch_psnr_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute average PSNR/SSIM over a batch of images.
    """
    assert preds.shape == targets.shape
    b = preds.shape[0]
    psnr_vals = []
    ssim_vals = []
    for i in range(b):
        psnr_vals.append(psnr(preds[i], targets[i]))
        ssim_vals.append(ssim(preds[i], targets[i]))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))

