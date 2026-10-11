import os
import shutil
import random

# ==============================
# SETTINGS
# ==============================

SOURCE_DIR = r"E:\FreshLife_AI\dataset\test"
OUTPUT_DIR = r"E:\FreshLife_AI\dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ==============================
# CHECK RATIOS
# ==============================

if TRAIN_RATIO + VAL_RATIO + TEST_RATIO != 1.0:
    raise ValueError("Train, validation and test ratios must add up to 1.0")

random.seed(RANDOM_SEED)

# ==============================
# CHECK SOURCE FOLDER
# ==============================

if not os.path.exists(SOURCE_DIR):
    raise FileNotFoundError(
        f"Source folder not found:\n{SOURCE_DIR}"
    )

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================

train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "validation")
test_dir = os.path.join(OUTPUT_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ==============================
# GET CLASS FOLDERS
# ==============================

classes = [
    folder for folder in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, folder))
]

if not classes:
    raise ValueError("No class folders found in source dataset.")

print("\nClasses found:")
for class_name in classes:
    print(" -", class_name)

# ==============================
# IMAGE EXTENSIONS
# ==============================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# ==============================
# SPLIT EACH CLASS
# ==============================

for class_name in classes:

    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    images = [
        file for file in os.listdir(source_class_dir)
        if file.lower().endswith(valid_extensions)
    ]

    if len(images) == 0:
        print(f"\nSkipping empty class: {class_name}")
        continue

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Create class folders
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # ==============================
    # COPY TRAIN IMAGES
    # ==============================

    for image in train_images:

        source = os.path.join(source_class_dir, image)
        destination = os.path.join(train_class_dir, image)

        shutil.copy2(source, destination)

    # ==============================
    # COPY VALIDATION IMAGES
    # ==============================

    for image in val_images:

        source = os.path.join(source_class_dir, image)
        destination = os.path.join(val_class_dir, image)

        shutil.copy2(source, destination)

    # ==============================
    # COPY TEST IMAGES
    # ==============================

    for image in test_images:

        source = os.path.join(source_class_dir, image)
        destination = os.path.join(test_class_dir, image)

        shutil.copy2(source, destination)

    print(
        f"\n{class_name}"
        f"\n  Total      : {total}"
        f"\n  Train      : {len(train_images)}"
        f"\n  Validation : {len(val_images)}"
        f"\n  Test       : {len(test_images)}"
    )

# ==============================
# COMPLETE
# ==============================

print("\n" + "=" * 50)
print("DATASET SPLIT COMPLETED SUCCESSFULLY")
print("=" * 50)

print("\nCreated dataset:")
print(OUTPUT_DIR)

print("\nStructure:")
print("dataset_split/")
print("├── train/")
print("├── validation/")
print("└── test/")