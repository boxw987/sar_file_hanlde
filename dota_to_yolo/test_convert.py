import os
import shutil
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import yaml # For reading/writing data.yaml

# --- Configuration ---
# Original dataset for testing
ORIGINAL_TEST_BASE_DIR = Path("D:/sl/SL-TEST/test") # Absolute path to your 'test' folder

# Main output directory where train/val (and now test) YOLO data resides/will reside
OUTPUT_YOLO_DATASET_DIR = Path("D:/sl/sl_yolo_dataset")

# --- Subdirectories in the original test dataset ---
ORIGINAL_TEST_IMAGES_DIR = ORIGINAL_TEST_BASE_DIR / "PNGImages"
ORIGINAL_TEST_ANNOTATIONS_DIR = ORIGINAL_TEST_BASE_DIR / "Annotations"

# --- Output subdirectories for YOLO format ---
# These paths are for data.yaml and for where test data will be placed.
# Train and Val paths are assumed to exist if data.yaml needs to be created from scratch.
YOLO_TRAIN_IMAGES_DIR = OUTPUT_YOLO_DATASET_DIR / "train" / "images"
YOLO_VAL_IMAGES_DIR = OUTPUT_YOLO_DATASET_DIR / "val" / "images"

YOLO_TEST_DIR = OUTPUT_YOLO_DATASET_DIR / "test"
YOLO_TEST_IMAGES_DIR = YOLO_TEST_DIR / "images"
YOLO_TEST_LABELS_DIR = YOLO_TEST_DIR / "labels"

# Path to the classes file (expected to exist from previous train/val conversion)
CLASSES_FILE_PATH = OUTPUT_YOLO_DATASET_DIR / "classes.txt"
DATA_YAML_PATH = OUTPUT_YOLO_DATASET_DIR / "data.yaml"


def get_image_dimensions(image_path):
    """Gets width and height of an image."""
    with Image.open(image_path) as img:
        return img.width, img.height

def convert_dota_to_yolo_obb(dota_annotation_path, image_width, image_height, class_to_id_map):
    """
    Converts a single DOTA annotation file to YOLO OBB format lines.
    """
    yolo_lines = []
    if not dota_annotation_path.exists():
        print(f"Warning: Annotation file not found: {dota_annotation_path}")
        return yolo_lines

    with open(dota_annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if "imagesource" in line.lower() or "gsd" in line.lower(): # Skip DOTA header lines
                continue
            if len(parts) < 9:
                print(f"Warning: Malformed line in {dota_annotation_path}: {line.strip()} (parts: {len(parts)})")
                continue
            try:
                coords = [float(c) for c in parts[:8]]
                class_name = parts[8]

                if class_name not in class_to_id_map:
                    print(f"Warning: Unknown class '{class_name}' in {dota_annotation_path}. Skipping this object. Known classes: {list(class_to_id_map.keys())}")
                    continue
                class_id = class_to_id_map[class_name]

                normalized_coords = []
                for i in range(0, 8, 2):
                    norm_x = coords[i] / image_width
                    norm_y = coords[i+1] / image_height
                    normalized_coords.append(max(0.0, min(1.0, norm_x)))
                    normalized_coords.append(max(0.0, min(1.0, norm_y)))
                yolo_line = f"{class_id} {' '.join(map(str, normalized_coords))}"
                yolo_lines.append(yolo_line)
            except ValueError:
                print(f"Warning: Could not parse coordinates in {dota_annotation_path}: {line.strip()}")
            except IndexError:
                print(f"Warning: Index error parsing line in {dota_annotation_path}: {line.strip()}")
    return yolo_lines

def process_test_files(image_file_list, source_annotations_dir, dest_img_dir, dest_label_dir, class_to_id_map):
    """
    Processes test image files: copies them and converts their annotations.
    """
    print(f"\nProcessing test set...")
    for img_path in tqdm(image_file_list, desc="Converting test set"):
        base_name = img_path.stem
        dest_image_path = dest_img_dir / img_path.name
        shutil.copy2(img_path, dest_image_path)

        try:
            img_width, img_height = get_image_dimensions(img_path)
        except Exception as e:
            print(f"Error getting dimensions for {img_path}: {e}. Skipping this image.")
            if dest_image_path.exists():
                dest_image_path.unlink()
            continue

        original_ann_path = source_annotations_dir / (base_name + ".txt")
        yolo_obb_lines = convert_dota_to_yolo_obb(original_ann_path, img_width, img_height, class_to_id_map)

        dest_label_path = dest_label_dir / (base_name + ".txt")
        with open(dest_label_path, 'w', encoding='utf-8') as f_out:
            for line in yolo_obb_lines:
                f_out.write(line + "\n")
    print(f"Test set processing complete.")


def main():
    print(f"Original TEST dataset base: {ORIGINAL_TEST_BASE_DIR}")
    print(f"Output YOLO dataset main directory: {OUTPUT_YOLO_DATASET_DIR}")

    # 0. Validate original test paths
    if not ORIGINAL_TEST_IMAGES_DIR.exists() or not ORIGINAL_TEST_ANNOTATIONS_DIR.exists():
        print(f"Error: Original TEST image directory ({ORIGINAL_TEST_IMAGES_DIR}) or "
              f"annotations directory ({ORIGINAL_TEST_ANNOTATIONS_DIR}) not found.")
        return

    # 1. Create output directories for the test set
    YOLO_TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_TEST_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directories for test set created/ensured: {YOLO_TEST_IMAGES_DIR}, {YOLO_TEST_LABELS_DIR}")

    # 2. Load class names and create class_to_id mapping
    # This relies on classes.txt being present from the train/val conversion.
    if not CLASSES_FILE_PATH.exists():
        print(f"Error: '{CLASSES_FILE_PATH}' not found.")
        print("This file is required to map class names to IDs and should have been generated during train/val conversion.")
        print("Please ensure it exists in the output YOLO dataset directory.")
        return

    with open(CLASSES_FILE_PATH, 'r', encoding='utf-8') as f:
        sorted_class_names = [line.strip() for line in f if line.strip()]

    if not sorted_class_names:
        print(f"Error: No class names found in '{CLASSES_FILE_PATH}'.")
        return

    class_to_id = {name: i for i, name in enumerate(sorted_class_names)}
    nc = len(sorted_class_names)

    print("\nUsing Class to ID mapping from classes.txt:")
    for name, idx in class_to_id.items():
        print(f"- {name}: {idx}")
    print(f"Number of classes (nc): {nc}")

    # 3. Process TEST files
    all_test_image_files = [f for f in ORIGINAL_TEST_IMAGES_DIR.glob("*.png") if f.is_file()]
    if not all_test_image_files:
        print(f"No PNG images found in {ORIGINAL_TEST_IMAGES_DIR}. Skipping test set processing.")
    else:
        print(f"\nFound {len(all_test_image_files)} images in the original test set.")
        process_test_files(all_test_image_files, ORIGINAL_TEST_ANNOTATIONS_DIR,
                             YOLO_TEST_IMAGES_DIR, YOLO_TEST_LABELS_DIR, class_to_id)

    # 4. Update or Create data.yaml
    print(f"\nUpdating/Creating {DATA_YAML_PATH}...")
    yaml_data = {}
    if DATA_YAML_PATH.exists():
        try:
            with open(DATA_YAML_PATH, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data is None: # Handle empty or invalid YAML file
                    yaml_data = {}
                print("Loaded existing data.yaml.")
        except yaml.YAMLError as e:
            print(f"Warning: Could not parse existing data.yaml: {e}. A new one will be created.")
            yaml_data = {}
        except Exception as e:
            print(f"Warning: Error reading existing data.yaml: {e}. A new one will be created.")
            yaml_data = {}


    # Ensure paths are absolute and in string format for YAML
    # If train/val paths are not in existing yaml, add them assuming standard structure
    if 'train' not in yaml_data or not yaml_data['train']:
        yaml_data['train'] = str(YOLO_TRAIN_IMAGES_DIR.resolve())
        print(f"Setting 'train' path in data.yaml: {yaml_data['train']}")

    if 'val' not in yaml_data or not yaml_data['val']:
        yaml_data['val'] = str(YOLO_VAL_IMAGES_DIR.resolve())
        print(f"Setting 'val' path in data.yaml: {yaml_data['val']}")
    
    yaml_data['test'] = str(YOLO_TEST_IMAGES_DIR.resolve())
    yaml_data['nc'] = nc
    yaml_data['names'] = sorted_class_names # Use list of names directly

    try:
        with open(DATA_YAML_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
        print(f"Successfully updated/created {DATA_YAML_PATH}")
    except Exception as e:
        print(f"Error writing data.yaml: {e}")


    print("\nTest set conversion and data.yaml update complete!")
    print(f"YOLO formatted test set saved to: {YOLO_TEST_DIR}")
    print(f"data.yaml updated at: {DATA_YAML_PATH}")

if __name__ == "__main__":
    # Ensure Pillow, tqdm, and PyYAML are installed:
    # pip install Pillow tqdm PyYAML
    main()
