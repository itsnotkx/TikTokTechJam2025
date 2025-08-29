import os

def collapse_classes_to_one(base_dir):
    """
    Collapse all YOLO classes into a single class '0'.
    
    Args:
        base_dir (str): Path to the dataset base dir, e.g. 'svhn_Data/data'
    """
    splits = ["train", "test", "valid"]
    label_base = os.path.join(base_dir, "labels")
    
    for split in splits:
        split_path = os.path.join(label_base, split)
        if not os.path.exists(split_path):
            continue
        
        for label_file in os.listdir(split_path):
            file_path = os.path.join(split_path, label_file)
            new_lines = []
            
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # skip malformed
                    # Replace class_id with 0
                    parts[0] = "1"
                    new_lines.append(" ".join(parts))
            
            # Overwrite file with collapsed labels
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n" if new_lines else "")
    
    print("✅ All classes collapsed into class 1 (numbers).")

# Usage
collapse_classes_to_one("svhn_data/data")
