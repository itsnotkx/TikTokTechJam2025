import os
import shutil

def convert_to_yolo_format(src_dir, dest_dir):
    """
    Convert dataset from format:
        src_dir/
            train/images, train/labels
            test/images, test/labels
            valid/images, valid/labels
    To YOLO format:
        dest_dir/
            images/train, images/test, images/valid
            labels/train, labels/test, labels/valid
    """
    
    splits = ["train", "test", "valid"]
    
    for split in splits:
        # Define source folders
        img_src = os.path.join(src_dir, split, "images")
        lbl_src = os.path.join(src_dir, split, "labels")
        
        # Define destination folders
        img_dest = os.path.join(dest_dir, "images", split)
        lbl_dest = os.path.join(dest_dir, "labels", split)
        
        # Create destination folders if they don't exist
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        # Copy images
        if os.path.exists(img_src):
            for f in os.listdir(img_src):
                shutil.copy(os.path.join(img_src, f), img_dest)
        
        # Copy labels
        if os.path.exists(lbl_src):
            for f in os.listdir(lbl_src):
                shutil.copy(os.path.join(lbl_src, f), lbl_dest)

    print(f"Dataset successfully converted to YOLO format at: {dest_dir}")

def main():
    convert_to_yolo_format("licenseplate_data", "licenseplate_yolo")
    
if __name__ == "__main__":
    main()