import os
import shutil
import random
from pathlib import Path

SOURCE = r"C:\Users\praktikan\.cache\kagglehub\datasets\dionisiusdh\indonesian-batik-motifs\versions\1"
TARGET = "dataset"

TRAIN_RATIO = 0.8

random.seed(42)

os.makedirs(os.path.join(TARGET, "train"), exist_ok=True)
os.makedirs(os.path.join(TARGET, "val"), exist_ok=True)

for class_dir in os.listdir(SOURCE):
    src_class = os.path.join(SOURCE, class_dir)

    if not os.path.isdir(src_class):
        continue

    images = []

    for file in os.listdir(src_class):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            images.append(file)

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    train_target = os.path.join(TARGET, "train", class_dir)
    val_target = os.path.join(TARGET, "val", class_dir)

    os.makedirs(train_target, exist_ok=True)
    os.makedirs(val_target, exist_ok=True)

    for img in train_imgs:
        shutil.copy2(
            os.path.join(src_class, img),
            os.path.join(train_target, img)
        )

    for img in val_imgs:
        shutil.copy2(
            os.path.join(src_class, img),
            os.path.join(val_target, img)
        )

    print(f"{class_dir}: {len(train_imgs)} train | {len(val_imgs)} val")

print("\nDataset siap!")