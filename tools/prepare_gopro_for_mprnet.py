import os
import shutil

SRC_ROOT = "D:/Gopro"

# Destination inside your project
DST_ROOT = "MPRNet/Deblurring/Datasets/GoPro"


def prepare(split):
    print(f"Processing {split} set...")

    input_dir = os.path.join(DST_ROOT, split, "input")
    target_dir = os.path.join(DST_ROOT, split, "target")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    split_path = os.path.join(SRC_ROOT, split)

    blur_path = os.path.join(split_path, "blur")
    sharp_path = os.path.join(split_path, "sharp")

    if not os.path.isdir(blur_path) or not os.path.isdir(sharp_path):
        raise FileNotFoundError(f"Missing blur/sharp folders in {split_path}")

    blur_files = sorted([
        f for f in os.listdir(blur_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    count = 0

    for fname in blur_files:
        src_blur = os.path.join(blur_path, fname)
        src_sharp = os.path.join(sharp_path, fname)

        if not os.path.exists(src_sharp):
            print(f"⚠️ Skipping (no match): {fname}")
            continue

        dst_blur = os.path.join(input_dir, fname)
        dst_sharp = os.path.join(target_dir, fname)

        if not os.path.exists(dst_blur):
            shutil.copy(src_blur, dst_blur)
        if not os.path.exists(dst_sharp):
            shutil.copy(src_sharp, dst_sharp)

        count += 1

        if count % 500 == 0:
            print(f"Copied {count} images...")

    print(f"✅ Done {split}: {count} pairs copied\n")


if __name__ == "__main__":
    prepare("train")
    prepare("test")
    print("🎉 Dataset prepared for MPRNet!")