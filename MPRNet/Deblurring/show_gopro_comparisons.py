import os
import random
from glob import glob
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from datetime import datetime

INPUT_DIR = "Datasets/GoPro/test/input"
PRED_DIR = "checkpoints/Deblurring/results/GoPro"
GT_DIR = "Datasets/GoPro/test/target"
OUTPUT_DIR = "comparison_results"

NUM_IMAGES = 2


def load_image(path):
    return np.array(Image.open(path).convert("RGB")) / 255.0


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def compute_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2, data_range=1.0)


def find_common_files():
    input_files = {os.path.basename(p): p for p in glob(os.path.join(INPUT_DIR, "*"))}
    pred_files = {os.path.basename(p): p for p in glob(os.path.join(PRED_DIR, "*"))}
    gt_files = {os.path.basename(p): p for p in glob(os.path.join(GT_DIR, "*"))}

    common = sorted(set(input_files) & set(pred_files) & set(gt_files))
    return [(input_files[f], pred_files[f], gt_files[f], f) for f in common]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pairs = find_common_files()
    if not pairs:
        raise RuntimeError("No matching files found.")

    random.seed(time.time())
    pairs = random.sample(pairs, min(NUM_IMAGES, len(pairs)))

    rows = len(pairs)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (input_path, pred_path, gt_path, name) in enumerate(pairs):
        blurred = load_image(input_path)
        pred = load_image(pred_path)
        gt = load_image(gt_path)

        # Compute metrics
        psnr_val = compute_psnr(gt, pred)
        ssim_val = compute_ssim(gt, pred)

        # Plot images
        axes[i, 0].imshow(blurred)
        axes[i, 0].set_title("Blurred")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred)
        axes[i, 1].set_title(f"MPRNet\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(gt)
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(OUTPUT_DIR, f"mprnet_comparisons_{timestamp}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved comparison to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()