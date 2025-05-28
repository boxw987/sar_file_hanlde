import os
import shutil
import random
from PIL import Image # Pillow library for image dimensions
from pathlib import Path
from tqdm import tqdm # For progress bars

# --- Configuration ---
ORIGINAL_DATASET_BASE_DIR = Path("D:/sl/SL-TRAINVAL/trainval") # Absolute path to your 'trainval' folder
OUTPUT_YOLO_DATASET_DIR = Path("D:/sl/sl_yolo_dataset")    # Absolute path for the new YOLO formatted dataset

# --- Subdirectories in the original dataset ---
ORIGINAL_IMAGES_DIR = ORIGINAL_DATASET_BASE_DIR / "PNGImages"
ORIGINAL_ANNOTATIONS_DIR = ORIGINAL_DATASET_BASE_DIR / "Annotations"

# --- Output subdirectories for YOLO format ---
YOLO_TRAIN_DIR = OUTPUT_YOLO_DATASET_DIR / "train"
YOLO_VAL_DIR = OUTPUT_YOLO_DATASET_DIR / "val"

YOLO_TRAIN_IMAGES_DIR = YOLO_TRAIN_DIR / "images"
YOLO_TRAIN_LABELS_DIR = YOLO_TRAIN_DIR / "labels"
YOLO_VAL_IMAGES_DIR = YOLO_VAL_DIR / "images"
YOLO_VAL_LABELS_DIR = YOLO_VAL_DIR / "labels"

TRAIN_RATIO = 0.9

def get_image_dimensions(image_path):
    """Gets width and height of an image."""
    with Image.open(image_path) as img:
        return img.width, img.height

def convert_dota_to_yolo_obb(dota_annotation_path, image_width, image_height, class_to_id_map):
    """
    Converts a single DOTA annotation file to YOLO OBB format lines.
    DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    YOLO OBB format: class_index x1_norm y1_norm x2_norm y2_norm x3_norm y3_norm x4_norm y4_norm
    """
    yolo_lines = []
    if not dota_annotation_path.exists():
        print(f"Warning: Annotation file not found: {dota_annotation_path}")
        return yolo_lines

    with open(dota_annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: # Should be at least 8 coords + class_name (+ difficulty)
                print(f"Warning: Malformed line in {dota_annotation_path}: {line.strip()}")
                continue

            try:
                coords = [float(c) for c in parts[:8]]
                class_name = parts[8]
                # difficulty = parts[9] # We don't use difficulty in YOLO format

                if class_name not in class_to_id_map:
                    print(f"Warning: Unknown class '{class_name}' in {dota_annotation_path}. Skipping this object.")
                    continue

                class_id = class_to_id_map[class_name]

                # Normalize coordinates
                normalized_coords = []
                for i in range(0, 8, 2): # x1, y1, x2, y2...
                    norm_x = coords[i] / image_width
                    norm_y = coords[i+1] / image_height
                    # Clamp values to be within [0, 1] as sometimes annotations can be slightly outside
                    normalized_coords.append(max(0.0, min(1.0, norm_x)))
                    normalized_coords.append(max(0.0, min(1.0, norm_y)))

                yolo_line = f"{class_id} {' '.join(map(str, normalized_coords))}"
                yolo_lines.append(yolo_line)
            except ValueError:
                print(f"Warning: Could not parse coordinates in {dota_annotation_path}: {line.strip()}")
            except IndexError:
                print(f"Warning: Index error parsing line in {dota_annotation_path}: {line.strip()}")


    return yolo_lines

def main():
    print(f"Original dataset base: {ORIGINAL_DATASET_BASE_DIR}")
    print(f"Output YOLO dataset to: {OUTPUT_YOLO_DATASET_DIR}")

    if not ORIGINAL_IMAGES_DIR.exists() or not ORIGINAL_ANNOTATIONS_DIR.exists():
        print(f"Error: Original image directory ({ORIGINAL_IMAGES_DIR}) or "
              f"annotations directory ({ORIGINAL_ANNOTATIONS_DIR}) not found.")
        return

    # 1. Create output directories
    for dir_path in [YOLO_TRAIN_IMAGES_DIR, YOLO_TRAIN_LABELS_DIR, YOLO_VAL_IMAGES_DIR, YOLO_VAL_LABELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("Output directories created.")

    # 2. Discover all unique class names to create a mapping
    print("Discovering class names...")
    all_class_names = set()
    for ann_file in tqdm(list(ORIGINAL_ANNOTATIONS_DIR.glob("*.txt")), desc="Scanning annotations for classes"):
        if not ann_file.is_file():
            continue
        try:
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9: # 8 coords + class_name
                        all_class_names.add(parts[8])
        except Exception as e:
            print(f"Error reading {ann_file}: {e}")


    if not all_class_names:
        print("Error: No class names found in annotations. Please check your annotation files.")
        return

    sorted_class_names = sorted(list(all_class_names))
    class_to_id = {name: i for i, name in enumerate(sorted_class_names)}

    print("\nClass to ID mapping:")
    for name, idx in class_to_id.items():
        print(f"- {name}: {idx}")

    # Save a classes.txt or data.yaml for YOLO (optional but good practice)
    with open(OUTPUT_YOLO_DATASET_DIR / "classes.txt", 'w') as f:
        for name in sorted_class_names:
            f.write(f"{name}\n")
    print(f"\nSaved classes.txt to {OUTPUT_YOLO_DATASET_DIR / 'classes.txt'}")

    # Create data.yaml (useful for training with Ultralytics YOLO)
    data_yaml_content = f"""
train: {YOLO_TRAIN_IMAGES_DIR.resolve()}
val: {YOLO_VAL_IMAGES_DIR.resolve()}

nc: {len(sorted_class_names)}
names: {sorted_class_names}
"""
    with open(OUTPUT_YOLO_DATASET_DIR / "data.yaml", 'w') as f:
        f.write(data_yaml_content)
    print(f"Saved data.yaml to {OUTPUT_YOLO_DATASET_DIR / 'data.yaml'}")


    # 3. List all images and shuffle for splitting
    all_image_files = [f for f in ORIGINAL_IMAGES_DIR.glob("*.png")] # Assuming PNG
    random.shuffle(all_image_files)
    num_images = len(all_image_files)
    num_train = int(num_images * TRAIN_RATIO)

    train_files = all_image_files[:num_train]
    val_files = all_image_files[num_train:]

    print(f"\nTotal images: {num_images}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    # 4. Process files: copy images and convert annotations
    def process_files(file_list, dest_img_dir, dest_label_dir, set_name):
        print(f"\nProcessing {set_name} set...")
        for img_path in tqdm(file_list, desc=f"Converting {set_name} set"):
            base_name = img_path.stem  # Filename without extension
            
            # Copy image
            dest_image_path = dest_img_dir / img_path.name
            shutil.copy2(img_path, dest_image_path)

            # Get image dimensions
            try:
                img_width, img_height = get_image_dimensions(img_path)
            except Exception as e:
                print(f"Error getting dimensions for {img_path}: {e}. Skipping this image.")
                # Optionally remove the copied image if dimensions can't be read
                if dest_image_path.exists():
                    dest_image_path.unlink()
                continue


            # Convert annotation
            original_ann_path = ORIGINAL_ANNOTATIONS_DIR / (base_name + ".txt")
            yolo_obb_lines = convert_dota_to_yolo_obb(original_ann_path, img_width, img_height, class_to_id)

            # Write YOLO label file
            dest_label_path = dest_label_dir / (base_name + ".txt")
            with open(dest_label_path, 'w') as f_out:
                for line in yolo_obb_lines:
                    f_out.write(line + "\n")
        print(f"{set_name} set processing complete.")

    # Process training set
    process_files(train_files, YOLO_TRAIN_IMAGES_DIR, YOLO_TRAIN_LABELS_DIR, "train")

    # Process validation set
    process_files(val_files, YOLO_VAL_IMAGES_DIR, YOLO_VAL_LABELS_DIR, "val")

    print("\nDataset conversion complete!")
    print(f"YOLO formatted dataset saved to: {OUTPUT_YOLO_DATASET_DIR}")

if __name__ == "__main__":
    # Ensure Pillow and tqdm are installed:
    # pip install Pillow tqdm
    main()