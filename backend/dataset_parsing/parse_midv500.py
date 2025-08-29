import os
import json
import shutil
from pathlib import Path

# Paths
root = Path("midv500_data")
coco_json = root / "midv500_coco.json"
output_dir = Path("midv500_yolo")

# Create YOLO folder structure
for split in ["train", "val"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Load COCO json
with open(coco_json, "r") as f:
    coco = json.load(f)

# We map everything to one class
class_map = {cat["id"]: 0 for cat in coco["categories"]}  # all -> 0

# Train/val split ratio
train_ratio = 0.8

# Group annotations per image
annotations_per_image = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    if img_id not in annotations_per_image:
        annotations_per_image[img_id] = []
    annotations_per_image[img_id].append(ann)

# Process images
images = coco["images"]
split_index = int(len(images) * train_ratio)

for i, img in enumerate(images):
    file_name = img["file_name"]
    width, height = img["width"], img["height"]
    img_id = img["id"]

    split = "train" if i < split_index else "val"

    # Copy image to YOLO dir
    src_path = root / file_name
    dst_path = output_dir / "images" / split / Path(file_name).name
    shutil.copy(src_path, dst_path)

    # Create label file
    label_path = output_dir / "labels" / split / (Path(file_name).stem + ".txt")
    with open(label_path, "w") as lf:
        if img_id in annotations_per_image:
            for ann in annotations_per_image[img_id]:
                bbox = ann["bbox"]  # COCO format: [x,y,w,h]
                x, y, w, h = bbox
                # Convert to YOLO format (normalized)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w /= width
                h /= height

                lf.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("✅ Conversion complete. YOLO dataset saved in:", output_dir)
