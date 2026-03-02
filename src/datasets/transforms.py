import random
from typing import Tuple

import torch
from torchvision import transforms


def get_basic_transform() -> transforms.Compose:
    """
    Convert PIL image -> tensor in [0, 1].
    """
    return transforms.ToTensor()


def random_crop_coords(h: int, w: int, crop_size: int) -> Tuple[int, int, int, int]:
    if h < crop_size or w < crop_size:
        # fall back to center crop if image smaller than crop
        top = max((h - crop_size) // 2, 0)
        left = max((w - crop_size) // 2, 0)
    else:
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
    bottom = min(top + crop_size, h)
    right = min(left + crop_size, w)
    return top, bottom, left, right


def paired_random_crop(
    img_blur: torch.Tensor,
    img_sharp: torch.Tensor,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the same random crop to both blurry and sharp tensors.

    Expects tensors of shape (C, H, W) with values in [0, 1].
    """
    _, h, w = img_blur.shape
    top, bottom, left, right = random_crop_coords(h, w, crop_size)
    img_blur = img_blur[:, top:bottom, left:right]
    img_sharp = img_sharp[:, top:bottom, left:right]
    return img_blur, img_sharp


def paired_random_flip(
    img_blur: torch.Tensor,
    img_sharp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the same random horizontal/vertical flip to both tensors.
    """
    if random.random() < 0.5:
        img_blur = torch.flip(img_blur, dims=[2])
        img_sharp = torch.flip(img_sharp, dims=[2])
    if random.random() < 0.5:
        img_blur = torch.flip(img_blur, dims=[1])
        img_sharp = torch.flip(img_sharp, dims=[1])
    return img_blur, img_sharp

