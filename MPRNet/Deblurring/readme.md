# MPRNet Deblurring Guide

## Overview
This guide explains how to prepare the GoPro dataset, train MPRNet, generate deblurred test images, evaluate PSNR/SSIM, and create visual comparisons.

## Project Structure
Use the `Deblurring` folder inside `MPRNet`.

Expected important files and folders:

- `train.py`
- `test.py`
- `training.yml`
- `evaluate_gopro.py`
- `show_gopro_comparisons.py`
- `Datasets/GoPro/`
- `checkpoints/`
- `eval_results/`

Dependencies:
pip install torch torchvision
pip install numpy<2
pip install matplotlib scikit-image pillow tqdm natsort opencv-python

## 1. Prepare the GoPro Dataset

### Expected layout
Inside `MPRNet/Deblurring`, the dataset should look like:

```text
Datasets/
  GoPro/
    train/
      input/
      target/
    test/
      input/
      target/
```

Where:

- `input/` = blurred images
- `target/` = sharp ground-truth images

### If your raw dataset is outside the repo
If your raw GoPro dataset is stored somewhere else, convert/copy it into the layout above before training or use
tools/prepare_gopro_for_mprnet.py

### Quick check
Make sure:

- blurred and target images have matching filenames
- train and test folders both exist
- image sizes are valid and readable

## 2. Configure Training

Edit `training.yml`.

### Recommended fine-tuning config
```yaml
GPU: [0]

VERBOSE: True

MODEL:
  MODE: 'Deblurring'
  SESSION: 'MPRNet'

OPTIM:
  BATCH_SIZE: 1
  NUM_EPOCHS: 5
  LR_INITIAL: 5e-5
  LR_MIN: 1e-6

TRAINING:
  VAL_AFTER_EVERY: 5
  RESUME: False
  TRAIN_PS: 256
  VAL_PS: 128
  TRAIN_DIR: './Datasets/GoPro/train'
  VAL_DIR: './Datasets/GoPro/test'
  SAVE_DIR: './checkpoints'
```

### What these settings mean
- `BATCH_SIZE`: number of patches processed at once
- `NUM_EPOCHS`: number of training epochs
- `LR_INITIAL`: initial learning rate
- `TRAIN_PS`: training patch size
- `VAL_PS`: validation patch size
- `VAL_AFTER_EVERY`: validation frequency

## 3. Pretrained Weights for Fine-Tuning
If you are fine-tuning instead of training from scratch, make sure `train.py` points to your pretrained checkpoint.

Example:

```python
pretrained_path = "D:/Gopro/model_deblurring.pth"
```

If the file exists, training will load those weights before fine-tuning.

## 4. Train the Model
Run from inside `MPRNet/Deblurring`:

```powershell
python train.py
```

### During training
You should see:

- training progress per epoch
- validation when enabled
- PSNR output after validation
- checkpoints being saved

### Saved checkpoints
Models are typically saved in:

```text
checkpoints/Deblurring/models/MPRNet/
```

Important files:

- `model_best.pth`
- `model_latest.pth`
- `model_epoch_X.pth`

Use `model_best.pth` for testing/evaluation.

## 5. Generate Deblurred Test Images
After training, run:

```powershell
python test.py
```

### Make sure `test.py` uses the correct checkpoint
It should point to:

```python
parser.add_argument('--weights', default='./checkpoints/Deblurring/models/MPRNet/model_best.pth', type=str)
```

### Make sure results are saved to
```python
parser.add_argument('--result_dir', default='./checkpoints/Deblurring/results/', type=str)
```

### Output images
Deblurred results will be written to:

```text
checkpoints/Deblurring/results/GoPro/
```

## 6. Evaluate PSNR and SSIM
Run:

```powershell
python evaluate_gopro.py
```

### What it does
This script:

- loads predicted images from `checkpoints/Deblurring/results/GoPro`
- loads ground truth images from `Datasets/GoPro/test/target`
- computes average PSNR
- computes average SSIM
- saves results to `eval_results/`

### Example output
```text
GoPro PSNR: 31.3315
GoPro SSIM: 0.9202
Results saved to eval_results/gopro_results_YYYY-MM-DD_HH-MM-SS.csv
```

## 7. Show GoPro Comparisons
Run:

```powershell
python show_gopro_comparisons.py
```

### What it does
This script randomly selects a few test examples and shows:

- blurred input
- MPRNet deblurred output
- ground truth image

It also displays per-image:

- PSNR
- SSIM

### Output
A comparison image is saved in a folder such as:

```text
comparison_results/
```

Example filename:

```text
mprnet_comparisons_YYYY-MM-DD_HH-MM-SS.png
```

### Why this is useful
Use this figure in your report or presentation to visually compare:

- how blurry the input is
- how much detail MPRNet restored
- how close the output is to the ground truth

## 8. Typical Workflow
From `MPRNet/Deblurring`:

```powershell
python train.py
python test.py
python evaluate_gopro.py
python show_gopro_comparisons.py
```

## 9. Troubleshooting

### Validation is slow
Set in `training.yml`:

```yaml
VAL_AFTER_EVERY: 5
```

or evaluate separately after training.

### Windows multiprocessing errors
Set `num_workers=0` in `train.py` and `test.py`.

### No predicted images found
Make sure `test.py` ran successfully and saved outputs to:

```text
checkpoints/Deblurring/results/GoPro/
```

### Bad PSNR after a later epoch
If training becomes unstable, use a lower learning rate such as:

```yaml
LR_INITIAL: 5e-5
```

and evaluate `model_best.pth`, not `model_latest.pth`.

## 10. Recommended Final Files to Keep
You can keep this minimal deblurring setup:

- `train.py`
- `test.py`
- `training.yml`
- `evaluate_gopro.py`
- `show_gopro_comparisons.py`
- `Datasets/`
- `checkpoints/`
- `eval_results/`
- `utils/`
- `config.py`
- `data_RGB.py`
- `dataset_RGB.py`
- `losses.py`
- `MPRNet.py`

Optional files that can often be removed if unused:

- `train_OG.py`
- `evaluate_RealBlur.py`
- `evaluate_GOPRO_HIDE`
- `__pycache__/`

## 11. Notes
- Use `model_best.pth` for final testing.
- Keep filenames consistent across input, prediction, and target images.
- For fair reporting, mention your patch size, batch size, and number of fine-tuning epochs.

