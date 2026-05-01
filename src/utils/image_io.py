import os
from typing import List, Optional

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _tensor_to_numpy_hwc(t: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) tensor in [0, 1] to a (H, W, C) uint8 numpy array."""
    arr = t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def save_single_comparison(
    blur: torch.Tensor,
    sharp: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
    model_label: str = "UNet (L1+SSIM)",
    psnr_val: Optional[float] = None,
    ssim_val: Optional[float] = None,
) -> None:
    """
    Save a single side-by-side comparison image: Blurred | Prediction | Ground Truth.

    Args:
        blur:        (C, H, W) single blurred image tensor in [0, 1]
        sharp:       (C, H, W) single ground truth tensor in [0, 1]
        pred:        (C, H, W) single prediction tensor in [0, 1]
        save_path:   destination file path (.png)
        model_label: label shown above the prediction column
        psnr_val:    PSNR for this image (shown in prediction title)
        ssim_val:    SSIM for this image (shown in prediction title)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback: stack the three images side by side with torchvision
        row = torch.stack([blur, pred, sharp], dim=0)   # (3, C, H, W)
        from torchvision.utils import make_grid, save_image
        ensure_dir(os.path.dirname(save_path) or ".")
        save_image(make_grid(row, nrow=3, padding=2), save_path)
        return

    ensure_dir(os.path.dirname(save_path) or ".")

    blur_np  = _tensor_to_numpy_hwc(blur)
    pred_np  = _tensor_to_numpy_hwc(pred)
    sharp_np = _tensor_to_numpy_hwc(sharp)

    # Build the middle-column title
    if psnr_val is not None and ssim_val is not None:
        pred_title = f"{model_label}\nPSNR: {psnr_val:.2f} dB,  SSIM: {ssim_val:.3f}"
    else:
        pred_title = model_label

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, img, title in zip(
        axes,
        [blur_np, pred_np, sharp_np],
        ["Blurred", pred_title, "Ground Truth"],
    ):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_labeled_triplet_grid(
    blur: torch.Tensor,
    sharp: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
    n_display: int = 4,
    model_label: str = "UNet (L1+SSIM)",
    psnr_vals: Optional[List[float]] = None,
    ssim_vals: Optional[List[float]] = None,
) -> None:
    """
    Save a labeled comparison grid (multiple rows stacked).
    Kept for use during training validation panels.
    For evaluation outputs use save_single_comparison instead.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        save_triplet_grid(blur, sharp, pred, save_path, n_display)
        return

    ensure_dir(os.path.dirname(save_path) or ".")
    b = blur.shape[0]
    n = min(b, n_display)

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(15, 5 * n), squeeze=False)

    for i in range(n):
        blur_np  = _tensor_to_numpy_hwc(blur[i])
        pred_np  = _tensor_to_numpy_hwc(pred[i])
        sharp_np = _tensor_to_numpy_hwc(sharp[i])

        for col, (ax, img) in enumerate(zip(axes[i], [blur_np, pred_np, sharp_np])):
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                if col == 1 and psnr_vals is not None and ssim_vals is not None:
                    title = f"{model_label}\nPSNR: {psnr_vals[i]:.2f},  SSIM: {ssim_vals[i]:.3f}"
                elif col == 0:
                    title = "Blurred"
                elif col == 2:
                    title = "Ground Truth"
                else:
                    title = model_label
                ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
            elif col == 1 and psnr_vals is not None and ssim_vals is not None:
                ax.set_title(
                    f"PSNR: {psnr_vals[i]:.2f},  SSIM: {ssim_vals[i]:.3f}",
                    fontsize=12, pad=6,
                )

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_triplet_grid(
    blur: torch.Tensor,
    sharp: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
    n_display: int = 4,
) -> None:
    """
    Save a plain grid comparing blur / prediction / sharp (no text labels).
    Kept for backwards compatibility.

    Expects tensors with shape (B, C, H, W) in [0, 1].
    """
    ensure_dir(os.path.dirname(save_path) or ".")
    b = blur.shape[0]
    n = min(b, n_display)

    rows: List[torch.Tensor] = []
    for i in range(n):
        rows.extend([blur[i], pred[i], sharp[i]])

    grid = make_grid(rows, nrow=3, padding=2)
    save_image(grid, save_path)
