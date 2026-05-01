import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.datasets.gopro_dataset import GoProDataset
from src.models.unet import UNet
from src.utils.image_io import ensure_dir, save_single_comparison
from src.utils.metrics import fast_batch_psnr_ssim, per_image_psnr_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deblurring model on GoPro test set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("experiments", "unet64_l1_ssim_epoch30", "unet64_l1_ssim_epoch30_best.pth"),
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument("--gopro-root", type=str, default="data/GOPRO_Large")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size. Larger values are faster on MPS (default: 8).")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    default_device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--output-dir", type=str, default="outputs_eval")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="If >0, limit number of evaluated samples.")
    parser.add_argument("--save-samples", action="store_true", default=True,
                        help="Save one labeled comparison image per sample. Enabled by default.")
    parser.add_argument("--n-samples", type=int, default=40,
                        help="Number of individual comparison images to save (default: 40).")
    parser.add_argument("--model-label", type=str, default="UNet (L1+SSIM)",
                        help="Label shown above the prediction column in saved images.")
    return parser.parse_args()


def build_loader(
    gopro_root: str,
    batch_size: int,
    crop_size: int,
    num_workers: int,
    max_samples: int,
    pin_memory: bool,
) -> DataLoader:
    ds = GoProDataset(root=gopro_root, split="test", crop_size=crop_size, augment=False)

    if max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Keeps workers alive between iterations — big speedup on macOS
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(in_channels=3, out_channels=3, base_channels=64).to(device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
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
        gopro_root=args.gopro_root,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(args.checkpoint, device)
    print(f"Loaded  : {args.checkpoint}")
    print(f"Device  : {device}")
    print(f"Batches : {len(loader)}  (batch_size={args.batch_size})")

    total_psnr   = 0.0
    total_ssim   = 0.0
    n_batches    = 0
    images_saved = 0   # track how many individual comparison files we've written

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            blur  = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)
            pred  = model(blur)

            # ── Fast on-device metrics for every batch ─────────────────────
            bp, bs = fast_batch_psnr_ssim(pred, sharp)
            total_psnr += bp
            total_ssim += bs
            n_batches  += 1

            # ── One separate image file per sample ─────────────────────────
            # scikit-image SSIM only called for samples we're actually saving
            if args.save_samples and images_saved < args.n_samples:
                remaining = args.n_samples - images_saved
                n_save    = min(blur.size(0), remaining)

                psnr_vals, ssim_vals = per_image_psnr_ssim(
                    pred[:n_save], sharp[:n_save]
                )

                for j in range(n_save):
                    save_path = os.path.join(
                        samples_dir, f"sample_{images_saved + 1:04d}.png"
                    )
                    save_single_comparison(
                        blur[j], sharp[j], pred[j],
                        save_path,
                        model_label=args.model_label,
                        psnr_val=psnr_vals[j],
                        ssim_val=ssim_vals[j],
                    )
                    images_saved += 1

    if n_batches == 0:
        print("No samples evaluated.")
        return

    avg_psnr = total_psnr / n_batches
    avg_ssim = total_ssim / n_batches
    print(f"\nResults on GoPro test set:")
    print(f"  Average PSNR : {avg_psnr:.3f} dB")
    print(f"  Average SSIM : {avg_ssim:.4f}")
    if args.save_samples:
        print(f"\n{images_saved} comparison images saved to: {samples_dir}/")


if __name__ == "__main__":
    main()
