import argparse
import csv
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


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def batch_psnr_torch(preds: torch.Tensor, targets: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(preds, targets).item()
    if mse == 0:
        return 100.0
    return 20.0 * float(torch.log10(torch.tensor(max_val))) - 10.0 * float(torch.log10(torch.tensor(mse)))


def append_metrics_row(
    csv_path: str,
    epoch: int,
    split: str,
    total_loss: float,
    charb_loss: float,
    ssim_value: float,
    psnr: float,
    ssim: float,
) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "split", "total_loss", "charb_loss", "ssim_value", "psnr", "ssim"])
        writer.writerow([epoch, split, total_loss, charb_loss, ssim_value, psnr, ssim])


def save_curves_plots(metrics_csv: str, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as e:
        print(f"matplotlib not available, skipping plots: {e}")
        return

    rows: list[dict[str, str]] = []
    with open(metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def series(split: str, key: str) -> tuple[list[int], list[float]]:
        xs: list[int] = []
        ys: list[float] = []
        for r in rows:
            if r["split"] != split:
                continue
            xs.append(int(r["epoch"]))
            ys.append(float(r[key]))
        return xs, ys

    tr_x, tr_y = series("train", "total_loss")
    va_x, va_y = series("val", "total_loss")
    if tr_x and va_x:
        plt.figure()
        plt.plot(tr_x, tr_y, label="train")
        plt.plot(va_x, va_y, label="val")
        plt.xlabel("epoch")
        plt.ylabel("total_loss")
        plt.title("Total loss over epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_total_loss.png"))
        plt.close()

    va_x, va_psnr = series("val", "psnr")
    if va_x:
        plt.figure()
        plt.plot(va_x, va_psnr, label="val")
        plt.xlabel("epoch")
        plt.ylabel("psnr")
        plt.title("Validation PSNR over epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_val_psnr.png"))
        plt.close()

    va_x, va_ssim = series("val", "ssim")
    if va_x:
        plt.figure()
        plt.plot(va_x, va_ssim, label="val")
        plt.xlabel("epoch")
        plt.ylabel("ssim")
        plt.title("Validation SSIM over epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_val_ssim.png"))
        plt.close()


def save_fixed_val_panel(
    model: nn.Module,
    val_ds,
    device: torch.device,
    out_path: str,
    indices: list[int],
) -> None:
    model.eval()
    picked = [i for i in indices if 0 <= i < len(val_ds)]
    if not picked:
        return
    blur_list = []
    sharp_list = []
    with torch.no_grad():
        for idx in picked:
            sample = val_ds[idx]
            blur_list.append(sample["blur"])
            sharp_list.append(sample["sharp"])
        blur = torch.stack(blur_list, dim=0).to(device)
        sharp = torch.stack(sharp_list, dim=0).to(device)
        pred = model(blur)
    save_triplet_grid(blur, sharp, pred, out_path, n_display=len(picked))


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
    parser.add_argument("--output-dir", type=str, default=os.path.join("experiments", "unet64_charb_ssim_epoch30"))
    parser.add_argument("--loss", type=str, default="charb_ssim", choices=["l1_ssim", "charb_ssim"])
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
) -> tuple[DataLoader, DataLoader, torch.utils.data.Dataset, torch.utils.data.Dataset]:
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
    return train_loader, val_loader, train_ds, val_ds


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
) -> tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_charb_loss = 0.0
    total_ssim_value = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)
            pred = model(blur)
            ssim_value = float(differentiable_ssim(pred, sharp).item())
            ssim_loss_term = 1.0 - ssim_value
            if loss_type == "l1_ssim":
                charb_loss = float(F.l1_loss(pred, sharp).item())
            else:
                charb_loss = float(charbonnier_loss(pred, sharp).item())
            loss = 0.8 * charb_loss + 0.2 * ssim_loss_term
            psnr_val, ssim_val = batch_psnr_ssim(pred, sharp)
            total_loss += loss
            total_charb_loss += charb_loss
            total_ssim_value += ssim_value
            total_psnr += psnr_val
            total_ssim += ssim_val
            n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / n_batches,
        total_charb_loss / n_batches,
        total_ssim_value / n_batches,
        total_psnr / n_batches,
        total_ssim / n_batches,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ensure_dir(args.output_dir)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    samples_dir = os.path.join(args.output_dir, "samples")
    ensure_dir(checkpoints_dir)
    ensure_dir(samples_dir)
    metrics_csv = os.path.join(args.output_dir, "metrics.csv")

    print(f"Using device: {device}")

    train_loader, val_loader, _, val_ds = build_dataloaders(
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
        epoch_total_loss = 0.0
        epoch_charb_loss = 0.0
        epoch_ssim_value = 0.0
        epoch_psnr = 0.0
        n_train_batches = 0
        for batch in pbar:
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)

            optimizer.zero_grad()
            pred = model(blur)
            ssim_value = differentiable_ssim(pred, sharp)
            ssim_loss_term = 1.0 - ssim_value
            if args.loss == "l1_ssim":
                charb_loss = F.l1_loss(pred, sharp)
            else:
                # Experiment 4: U-Net + Charbonnier + SSIM
                charb_loss = charbonnier_loss(pred, sharp)
            # Experiment 4: combined Charbonnier + SSIM loss for better structural reconstruction
            loss = 0.8 * charb_loss + 0.2 * ssim_loss_term
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

            epoch_total_loss += float(loss.item())
            epoch_charb_loss += float(charb_loss.item())
            epoch_ssim_value += float(ssim_value.item())
            epoch_psnr += batch_psnr_torch(pred.detach(), sharp.detach())
            n_train_batches += 1

            global_step += 1

        if n_train_batches > 0:
            append_metrics_row(
                metrics_csv,
                epoch,
                "train",
                epoch_total_loss / n_train_batches,
                epoch_charb_loss / n_train_batches,
                epoch_ssim_value / n_train_batches,
                epoch_psnr / n_train_batches,
                epoch_ssim_value / n_train_batches,
            )

        # Validation
        if epoch % args.val_interval == 0:
            val_total_loss, val_charb_loss, val_ssim_value, val_psnr, val_ssim = validate(model, val_loader, device, args.loss)
            append_metrics_row(metrics_csv, epoch, "val", val_total_loss, val_charb_loss, val_ssim_value, val_psnr, val_ssim)
            print(f"[Epoch {epoch}] Val PSNR: {val_psnr:.3f}, SSIM: {val_ssim:.4f}, Loss: {val_total_loss:.4f}")

            # Deterministic qualitative comparison panel (fixed val indices)
            fixed_panel_path = os.path.join(samples_dir, f"val_fixed_epoch{epoch}.png")
            save_fixed_val_panel(model, val_ds, device, fixed_panel_path, indices=[0, 1, 2, 3, 4, 5, 6, 7])

            save_curves_plots(metrics_csv, args.output_dir)

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

