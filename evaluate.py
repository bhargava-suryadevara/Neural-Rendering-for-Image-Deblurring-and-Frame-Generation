import argparse
import os
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.datasets.gopro_dataset import GoProDataset
from src.datasets.realblur_dataset import RealBlurJDataset
from src.models.unet import UNet
from src.utils.image_io import ensure_dir, save_triplet_grid
from src.utils.metrics import batch_psnr_ssim


DatasetName = Literal["gopro", "realblur"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deblurring model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["gopro", "realblur"], required=True)
    parser.add_argument("--gopro-root", type=str, default="Data/GOPRO_Large")
    parser.add_argument("--realblur-root", type=str, default="Data/RealBlur")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs_eval")
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, limit number of evaluated samples.")
    parser.add_argument("--save-samples", action="store_true", help="Save example output images.")
    return parser.parse_args()


def build_loader(
    dataset_name: DatasetName,
    gopro_root: str,
    realblur_root: str,
    batch_size: int,
    crop_size: int,
    num_workers: int,
    max_samples: int,
) -> DataLoader:
    if dataset_name == "gopro":
        ds = GoProDataset(
            root=gopro_root,
            split="test",
            crop_size=crop_size,
            augment=False,
        )
    else:
        ds = RealBlurJDataset(
            realblur_root=realblur_root,
            split="test",
            crop_size=crop_size,
            augment=False,
        )

    if max_samples > 0:
        indices = list(range(min(max_samples, len(ds))))
        ds = Subset(ds, indices)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(in_channels=3, out_channels=3, base_channels=32).to(device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ensure_dir(args.output_dir)
    samples_dir = os.path.join(args.output_dir, "samples")
    if args.save_samples:
        ensure_dir(samples_dir)

    loader = build_loader(
        dataset_name=args.dataset,  # type: ignore[arg-type]
        gopro_root=args.gopro_root,
        realblur_root=args.realblur_root,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    model = load_model(args.checkpoint, device)
    print(f"Evaluating on {args.dataset} using device {device}")

    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)
            pred = model(blur)
            psnr_val, ssim_val = batch_psnr_ssim(pred, sharp)
            total_psnr += psnr_val
            total_ssim += ssim_val
            n_batches += 1

            if args.save_samples and i < 10:
                save_path = os.path.join(samples_dir, f"{args.dataset}_batch{i}.png")
                save_triplet_grid(blur, sharp, pred, save_path, n_display=min(4, blur.size(0)))

    if n_batches == 0:
        print("No samples evaluated.")
        return

    avg_psnr = total_psnr / n_batches
    avg_ssim = total_ssim / n_batches
    print(f"Average PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    main()

