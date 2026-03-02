import os
from typing import List, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import get_basic_transform, paired_random_crop, paired_random_flip


class GoProDataset(Dataset):
    """
    GoPro deblurring dataset.

    Expected structure:
        root/
          train/
            <scene>/
              blur/*.png
              sharp/*.png
          test/
            <scene>/
              blur/*.png
              sharp/*.png
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        assert split in {"train", "test"}
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.augment = augment
        self.to_tensor = get_basic_transform()

        self.pairs: List[Tuple[str, str]] = self._collect_pairs()

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        split_root = os.path.join(self.root, self.split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"GoPro split directory not found: {split_root}")

        scenes = sorted(
            d
            for d in os.listdir(split_root)
            if os.path.isdir(os.path.join(split_root, d))
        )
        for scene in scenes:
            scene_path = os.path.join(split_root, scene)
            blur_dir = os.path.join(scene_path, "blur")
            sharp_dir = os.path.join(scene_path, "sharp")
            if not (os.path.isdir(blur_dir) and os.path.isdir(sharp_dir)):
                continue

            blur_files = sorted(
                f
                for f in os.listdir(blur_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            for bf in blur_files:
                blur_path = os.path.join(blur_dir, bf)
                sharp_path = os.path.join(sharp_dir, bf)
                if os.path.isfile(sharp_path):
                    pairs.append((blur_path, sharp_path))

        if not pairs:
            raise RuntimeError(f"No GoPro image pairs found under {split_root}")
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        blur_path, sharp_path = self.pairs[idx]

        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        blur_tensor = self.to_tensor(blur_img)
        sharp_tensor = self.to_tensor(sharp_img)

        if self.crop_size is not None:
            blur_tensor, sharp_tensor = paired_random_crop(
                blur_tensor, sharp_tensor, self.crop_size
            )

        if self.augment:
            blur_tensor, sharp_tensor = paired_random_flip(blur_tensor, sharp_tensor)

        return {
            "blur": blur_tensor,
            "sharp": sharp_tensor,
            "blur_path": blur_path,
            "sharp_path": sharp_path,
        }

