import os
from typing import List

import torch
from torchvision.utils import make_grid, save_image


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_triplet_grid(
    blur: torch.Tensor,
    sharp: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
    n_display: int = 4,
) -> None:
    """
    Save a grid image comparing blur / prediction / sharp.

    Expects tensors with shape (B, C, H, W) in [0, 1].
    """
    ensure_dir(os.path.dirname(save_path))
    b = blur.shape[0]
    n = min(b, n_display)

    rows: List[torch.Tensor] = []
    for i in range(n):
        rows.extend([blur[i], pred[i], sharp[i]])

    grid = make_grid(rows, nrow=3, padding=2)
    save_image(grid, save_path)

