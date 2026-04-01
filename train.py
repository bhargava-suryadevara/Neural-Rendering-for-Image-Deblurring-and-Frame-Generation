import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from src.datasets.gopro_dataset import GoProDataset
from src.models.unet import UNet
from src.utils.image_io import ensure_dir, save_triplet_grid
from src.utils.metrics import batch_psnr_ssim
from src.utils.ssim_loss import differentiable_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deblurring model on GoPro.")
    parser.add_argument("--gopro-root", type=str, default="Data/GOPRO_Large")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--max-train-samples", type=int, default=0, help="If >0, limit number of training samples (sanity check).")
    parser.add_argument("--max-val-samples", type=int, default=0, help="If >0, limit number of val samples (sanity check).")
    parser.add_argument("--val-interval", type=int, default=1, help="Validate every N epochs.")
    return parser.parse_args()


def build_dataloaders(
    gopro_root: str,
    batch_size: int,
    crop_size: int,
    num_workers: int,
    pin_memory: bool,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_ds = GoProDataset(
        root=gopro_root,
        split="train",
        crop_size=crop_size,
        augment=True,
    )
    val_ds = GoProDataset(
        root=gopro_root,
        split="test",
        crop_size=crop_size,
        augment=False,
    )

    if max_train_samples > 0:
        indices = list(range(min(max_train_samples, len(train_ds))))
        train_ds = Subset(train_ds, indices)
    if max_val_samples > 0:
        indices = list(range(min(max_val_samples, len(val_ds))))
        val_ds = Subset(val_ds, indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)
            pred = model(blur)
            psnr_val, ssim_val = batch_psnr_ssim(pred, sharp)
            total_psnr += psnr_val
            total_ssim += ssim_val
            n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0
    return total_psnr / n_batches, total_ssim / n_batches


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ensure_dir(args.output_dir)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    samples_dir = os.path.join(args.output_dir, "samples")
    ensure_dir(checkpoints_dir)
    ensure_dir(samples_dir)

    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(
        gopro_root=args.gopro_root,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    # Experiment 1: increased UNet capacity for comparison with baseline
    model = UNet(in_channels=3, out_channels=3, base_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    best_psnr: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)

            optimizer.zero_grad()
            pred = model(blur)
            l1_loss = F.l1_loss(pred, sharp)
            ssim_value = differentiable_ssim(pred, sharp)
            ssim_loss_term = 1.0 - ssim_value
            # Experiment 3: combined L1 + SSIM loss for better structural reconstruction
            loss = 0.8 * l1_loss + 0.2 * ssim_loss_term
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

            # Save sample grid occasionally (first batch of epoch)
            if global_step % 200 == 0:
                sample_path = os.path.join(samples_dir, f"train_epoch{epoch}_step{global_step}.png")
                save_triplet_grid(blur, sharp, pred, sample_path, n_display=min(4, blur.size(0)))

            global_step += 1

        # Validation
        if epoch % args.val_interval == 0:
            val_psnr, val_ssim = validate(model, val_loader, device)
            print(f"[Epoch {epoch}] Val PSNR: {val_psnr:.3f}, SSIM: {val_ssim:.4f}")

            # Save latest checkpoint
            latest_ckpt = os.path.join(checkpoints_dir, "latest.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                },
                latest_ckpt,
            )

            # Save best checkpoint by PSNR
            if best_psnr is None or val_psnr > best_psnr:
                best_psnr = val_psnr
                best_ckpt = os.path.join(checkpoints_dir, "best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                    },
                    best_ckpt,
                )
                print(f"New best model saved with PSNR={val_psnr:.3f}")


if __name__ == "__main__":
    main()

