import os
import glob
import random
import shutil
import zipfile
from pathlib import Path
import yaml

ZIP_PATH = "Opensource dataset traffic.yolov8 (2).zip"

EXTRACT_DIR = "addis_traffic"
SPLIT_DIR = "addis_traffic_split"

CLASS_NAMES = [
    "bicycle",
    "bus",
    "bus_stop",
    "car",
    "crosswalk",
    "cyclist",
    "mini_bus_taxi",
    "motorcycle",
    "pedestrian",
    "road_sign",
    "three_wheeler",
    "traffic_light",
    "traffic_sign",
    "truck",
]

# Remove old folders
for folder in [EXTRACT_DIR, SPLIT_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# Extract ZIP
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Extracted dataset to:", EXTRACT_DIR)
print("Contents:", os.listdir(EXTRACT_DIR))

ORIGINAL_IMG_DIR = os.path.join(EXTRACT_DIR, "train", "images")
ORIGINAL_LBL_DIR = os.path.join(EXTRACT_DIR, "train", "labels")

# Create split folders
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(SPLIT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(SPLIT_DIR, split, "labels"), exist_ok=True)

# Get images
image_files = glob.glob(os.path.join(ORIGINAL_IMG_DIR, "*.*"))

random.seed(42)
random.shuffle(image_files)

n_total = len(image_files)
n_train = int(n_total * 0.70)
n_valid = int(n_total * 0.20)

train_files = image_files[:n_train]
valid_files = image_files[n_train:n_train + n_valid]
test_files = image_files[n_train + n_valid:]

def copy_image_and_label(files, split_name):
    for img_path in files:
        img_name = os.path.basename(img_path)
        stem = Path(img_name).stem

        label_path = os.path.join(ORIGINAL_LBL_DIR, stem + ".txt")

        new_img_path = os.path.join(SPLIT_DIR, split_name, "images", img_name)
        new_lbl_path = os.path.join(SPLIT_DIR, split_name, "labels", stem + ".txt")

        shutil.copy2(img_path, new_img_path)

        if os.path.exists(label_path):
            shutil.copy2(label_path, new_lbl_path)
        else:
            print("Missing label:", img_name)

copy_image_and_label(train_files, "train")
copy_image_and_label(valid_files, "valid")
copy_image_and_label(test_files, "test")

print("Split completed.")
print("Train:", len(train_files))
print("Valid:", len(valid_files))
print("Test:", len(test_files))

# Create data.yaml
data_yaml = {
    "path": os.path.abspath(SPLIT_DIR),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES,
}

with open("data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("Created data.yaml")
print(yaml.dump(data_yaml, default_flow_style=False))