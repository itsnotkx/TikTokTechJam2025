import os
import shutil
from pathlib import Path

# Paths to original datasets
datasets = {
    "licenseplate": "licenseplate_yolo",
    "svhn": "svhn_data/data",
    "midv500": "midv500_yolo"
}

# Final merged dataset path
merged_root = Path("merged_yolo")

splits = ["train", "val", "test"]
folders = ["images", "labels"]

# Create merged dataset structure
for folder in folders:
    for split in splits:
        (merged_root / folder / split).mkdir(parents=True, exist_ok=True)

# Function to copy files and avoid name collisions
def copy_dataset(src_root, prefix):
    for folder in folders:
        for split in splits:
            src_dir = Path(src_root) / folder / split
            if not src_dir.exists():
                continue
            for file in src_dir.glob("*.*"):
                ext = file.suffix
                new_name = f"{prefix}_{file.stem}{ext}"
                dst = merged_root / folder / split / new_name
                shutil.copy(file, dst)

# Merge all datasets with unique prefixes
copy_dataset(datasets["licenseplate"], "lp")
copy_dataset(datasets["svhn"], "svhn")
copy_dataset(datasets["midv500"], "midv")

print("✅ Datasets merged into:", merged_root)
