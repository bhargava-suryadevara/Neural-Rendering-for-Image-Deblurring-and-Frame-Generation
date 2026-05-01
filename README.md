# Neural Rendering for Image Deblurring and Frame Generation

A two-phase project for supervised image deblurring using deep learning, trained and evaluated on the **GoPro Large** dataset.

---

## Project Overview

| Phase | Model | Goal |
|-------|-------|------|
| Phase 1 | U-Net | Supervised image deblurring baseline |
| Phase 2 | MPRNet | State-of-the-art multi-patch deblurring |

---

## Phase 1 — U-Net Deblurring

### Architecture

A lightweight encoder-decoder U-Net with skip connections:

- **Encoder:** 3 stages of double convolution (Conv → BN → ReLU × 2) + MaxPool, doubling channels at each stage (64 → 128 → 256)
- **Bottleneck:** Double convolution at 512 channels
- **Decoder:** Transposed convolutions with skip concatenation, halving channels back
- **Output:** 1×1 convolution clamped to [0, 1]

### Dataset

**GoPro Large** — pairs of motion-blurred and sharp frames from a GoPro camera.

Expected directory structure:

```
data/
  GOPRO_Large/
    train/
      <scene>/
        blur/  *.png
        sharp/ *.png
    test/
      <scene>/
        blur/  *.png
        sharp/ *.png
```

### Loss Function

Combined loss for best structural reconstruction:

```
Loss = 0.8 × L1 + 0.2 × (1 − SSIM)
```

### Experiments & Results

All experiments use batch size 4, crop size 256×256, lr=1e-4.

| Experiment | Model | Epochs | Loss | PSNR (dB) | SSIM | Notes |
|---|---|---|---|---|---|---|
| Baseline | UNet-32 | 30 | L1 | 25.610 | 0.7627 | Initial baseline, CPU |
| UNet-64 | UNet-64 | 30 | L1 | 25.808 | 0.7625 | Increased capacity |
| UNet-64 longer | UNet-64 | 50 | L1 | 25.840 | 0.7697 | Longer training on MPS |
| UNet-64 + Charb+SSIM | UNet-64 | 30 | Charb+SSIM | 25.900 | 0.7726 | Combined structural loss |
| **UNet-64 + L1+SSIM** | **UNet-64** | **30** | **L1+SSIM** | **25.918** | **0.7734** | **Best result** ✓ |

### Phase 2 — MPRNet Reference

| Model | PSNR (dB) | SSIM |
|-------|-----------|------|
| MPRNet | ~31.55 | 0.942 |

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

---

## Training

Train the best model (UNet-64, L1+SSIM loss, 30 epochs):

```bash
python3 train.py \
  --gopro-root data/GOPRO_Large \
  --epochs 30 \
  --batch-size 4 \
  --crop-size 256 \
  --num-workers 4
```

Outputs saved to `experiments/unet64_l1_ssim_epoch30/`:

- `checkpoints/best.pth` — best checkpoint by validation PSNR
- `checkpoints/latest.pth` — most recent checkpoint
- `samples/val_fixed_epoch{N}.png` — labeled validation panel after each epoch
- `metrics.csv` — per-epoch train/val PSNR, SSIM, and loss

### Sanity Check (fast end-to-end test)

```bash
python3 train.py \
  --gopro-root data/GOPRO_Large \
  --epochs 1 \
  --batch-size 2 \
  --crop-size 256 \
  --max-train-samples 32 \
  --max-val-samples 16
```

---

## Evaluation

Evaluate the best checkpoint and save individual comparison images:

```bash
python3 evaluate.py \
  --checkpoint experiments/unet64_l1_ssim_epoch30/checkpoints/best.pth \
  --gopro-root data/GOPRO_Large
```

This prints average PSNR and SSIM on the GoPro test set and saves **40 individual comparison images** to `outputs_eval/samples/`. Each image shows:

```
Blurred  |  UNet (L1+SSIM) — PSNR / SSIM  |  Ground Truth
```

### Evaluation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `experiments/unet64_l1_ssim_epoch30/checkpoints/best.pth` | Model checkpoint path |
| `--gopro-root` | `data/GOPRO_Large` | Dataset root directory |
| `--n-samples` | `40` | Number of comparison images to save |
| `--output-dir` | `outputs_eval` | Where to save results |
| `--batch-size` | `8` | Inference batch size |
| `--max-samples` | `0` (all) | Limit test samples for a quick check |

---

## Project Structure

```
.
├── train.py                        # Training script
├── evaluate.py                     # Evaluation script
├── requirements.txt
├── data/
│   └── GOPRO_Large/                # Dataset (not tracked by git)
├── src/
│   ├── models/
│   │   └── unet.py                 # U-Net architecture
│   ├── datasets/
│   │   ├── gopro_dataset.py        # GoPro data loader
│   │   └── transforms.py           # Paired crop & flip augmentation
│   └── utils/
│       ├── metrics.py              # PSNR / SSIM (fast on-device + scikit-image)
│       ├── ssim_loss.py            # Differentiable SSIM for training
│       └── image_io.py             # Labeled comparison image saving
├── experiments/
│   ├── experiment_log.csv          # Summary of all experiment results
│   ├── baseline/                   # UNet-32 results
│   ├── unet64_epoch30/             # UNet-64 30-epoch results
│   ├── unet64_epoch50/             # UNet-64 50-epoch results
│   ├── unet64_l1_ssim_epoch30/     # Best model ✓
│   └── unet64_charb_ssim_epoch30/  # Charbonnier+SSIM ablation
├── outputs_eval/
│   └── samples/                    # Individual comparison images from evaluate.py
└── MPRNet/                         # Phase 2 reference implementation
    └── Deblurring/
```

---

## Requirements

```
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.24.0
pillow >= 10.0.0
opencv-python >= 4.8.0
scikit-image >= 0.20.0
tqdm >= 4.65.0
matplotlib
pyyaml >= 6.0.0
```
