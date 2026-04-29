import os
from glob import glob

from datetime import datetime
import numpy as np
from natsort import natsorted
from skimage import io
from skimage.metrics import structural_similarity
import csv

def compute_psnr(image_true, image_test, data_range=1.0):
    mse = np.mean((image_true - image_test) ** 2, dtype=np.float64)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((data_range ** 2) / mse)


def compute_ssim(image_true, image_test):
    return structural_similarity(
        image_true,
        image_test,
        channel_axis=2,
        data_range=1.0,
    )


def load_image(path):
    img = io.imread(path).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def main():
    pred_dir = "checkpoints/Deblurring/results/GoPro"
    gt_dir = os.path.join("Datasets", "GoPro", "test", "target")

    pred_list = natsorted(
        glob(os.path.join(pred_dir, "*.png")) + glob(os.path.join(pred_dir, "*.jpg"))
    )
    gt_list = natsorted(
        glob(os.path.join(gt_dir, "*.png")) + glob(os.path.join(gt_dir, "*.jpg"))
    )

    assert len(pred_list) != 0, f"No predicted images found in {pred_dir}"
    assert len(gt_list) != 0, f"No ground-truth images found in {gt_dir}"
    assert len(pred_list) == len(gt_list), (
        f"Mismatch: {len(pred_list)} predictions vs {len(gt_list)} targets"
    )

    psnr_scores = []
    ssim_scores = []

    for gt_path, pred_path in zip(gt_list, pred_list):
        gt_img = load_image(gt_path)
        pred_img = load_image(pred_path)

        if gt_img.shape != pred_img.shape:
            raise ValueError(
                f"Shape mismatch:\nGT: {gt_path} -> {gt_img.shape}\n"
                f"PRED: {pred_path} -> {pred_img.shape}"
            )

        psnr_scores.append(compute_psnr(gt_img, pred_img))
        ssim_scores.append(compute_ssim(gt_img, pred_img))

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)

    print(f"GoPro PSNR: {avg_psnr:.4f}")
    print(f"GoPro SSIM: {avg_ssim:.4f}")

    os.makedirs("eval_results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_file = f"eval_results/gopro_results_{timestamp}.csv"

    with open(output_file, "w") as f:
        writer = csv.writer(f)
        f.write(f"PSNR: {avg_psnr:.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()