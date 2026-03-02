import os
from typing import List, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import get_basic_transform, paired_random_crop, paired_random_flip


class RealBlurJDataset(Dataset):
    """
    RealBlur-J dataset loader using provided train/test list files.

    Expected structure (relative to realblur_root):
        RealBlur-J_ECC_IMCORR_centroid_itensity_ref/sceneXXX/{blur,gt}/*.png

    List files (in realblur_root):
        RealBlur_J_train_list.txt
        RealBlur_J_test_list.txt

    Each line in the list file:
        <gt_relative_path> <blur_relative_path>
    """

    def __init__(
        self,
        realblur_root: str,
        split: str = "train",
        crop_size: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        assert split in {"train", "test"}
        self.realblur_root = realblur_root
        self.split = split
        self.crop_size = crop_size
        self.augment = augment
        self.to_tensor = get_basic_transform()

        self.pairs: List[Tuple[str, str]] = self._collect_pairs()

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        if self.split == "train":
            list_name = "RealBlur_J_train_list.txt"
        else:
            list_name = "RealBlur_J_test_list.txt"
        list_path = os.path.join(self.realblur_root, list_name)
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"RealBlur list file not found: {list_path}")

        pairs: List[Tuple[str, str]] = []
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                gt_rel, blur_rel = parts
                gt_path = os.path.join(self.realblur_root, gt_rel)
                blur_path = os.path.join(self.realblur_root, blur_rel)
                if os.path.isfile(gt_path) and os.path.isfile(blur_path):
                    pairs.append((blur_path, gt_path))

        if not pairs:
            raise RuntimeError(f"No RealBlur-J pairs found from {list_path}")
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        blur_path, gt_path = self.pairs[idx]

        blur_img = Image.open(blur_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        blur_tensor = self.to_tensor(blur_img)
        gt_tensor = self.to_tensor(gt_img)

        if self.crop_size is not None:
            blur_tensor, gt_tensor = paired_random_crop(
                blur_tensor, gt_tensor, self.crop_size
            )

        if self.augment:
            blur_tensor, gt_tensor = paired_random_flip(blur_tensor, gt_tensor)

        return {
            "blur": blur_tensor,
            "sharp": gt_tensor,
            "blur_path": blur_path,
            "sharp_path": gt_path,
        }

