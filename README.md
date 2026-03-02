## Neural Rendering for Image Deblurring (Phase 1)

This repository provides a clean baseline for **supervised image deblurring** using the **GoPro** and **RealBlur-J** datasets.

### Datasets

- **GoPro**: `Data/GOPRO_Large/{train,test}/<scene>/{blur,sharp}/*.png`
- **RealBlur-J**: `Data/RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/sceneXXX/{blur,gt}/*.png`
  - Train/test splits defined by:
    - `Data/RealBlur/RealBlur_J_train_list.txt`
    - `Data/RealBlur/RealBlur_J_test_list.txt`

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### Training (GoPro baseline)

```bash
python3 train.py \
  --gopro-root "Data/GOPRO_Large" \
  --batch-size 4 \
  --crop-size 256 \
  --epochs 1 \
  --num-workers 4
```

This will:

- Train on GoPro train split.
- Periodically save sample deblurred images under `outputs/samples/`.
- Save model checkpoints under `outputs/checkpoints/`.

### Evaluation

**GoPro test:**

```bash
python3 evaluate.py \
  --checkpoint "outputs/checkpoints/latest.pth" \
  --dataset gopro \
  --gopro-root "Data/GOPRO_Large"
```

**RealBlur-J test:**

```bash
python3 evaluate.py \
  --checkpoint "outputs/checkpoints/latest.pth" \
  --dataset realblur \
  --realblur-root "Data/RealBlur"
```

### Sanity check (tiny subset)

To quickly verify the pipeline end‑to‑end:

```bash
python3 train.py \
  --gopro-root "Data/GOPRO_Large" \
  --batch-size 2 \
  --crop-size 256 \
  --epochs 1 \
  --max-train-samples 32 \
  --max-val-samples 16
```

This should run in a few minutes on GPU and produce sample outputs and PSNR/SSIM numbers.

