import os
import shutil
from pathlib import Path
import random

# Paths
yolo_dir = Path("midv500_yolo")
val_images = list((yolo_dir / "images" / "val").glob("*.tif"))  # adjust if png
test_ratio = 0.1  # take 50% of val as test

# Create test dirs
(yolo_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
(yolo_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)

# Shuffle and split
random.shuffle(val_images)
test_count = int(len(val_images) * test_ratio)
test_images = val_images[:test_count]

for img_path in test_images:
    label_path = yolo_dir / "labels" / "val" / (img_path.stem + ".txt")

    # Move image
    shutil.move(str(img_path), yolo_dir / "images" / "test" / img_path.name)

    # Move label
    if label_path.exists():
        shutil.move(str(label_path), yolo_dir / "labels" / "test" / label_path.name)

print(f"✅ Moved {test_count} images+labels into test split.")
