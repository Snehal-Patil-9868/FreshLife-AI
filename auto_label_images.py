import os
import json
import csv
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# FreshLife AI - Automatic Image Pseudo Labelling
# ============================================================

BASE_DIR = r"E:\FreshLife_AI"

IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fruit_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "auto_labeled_dataset")
CSV_PATH = os.path.join(BASE_DIR, "auto_labels.csv")

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.85

# ------------------------------------------------------------
# Create output folders
# ------------------------------------------------------------

LABELS = [
    "Fresh",
    "Spoilage"
]

for label in LABELS:
    os.makedirs(
        os.path.join(OUTPUT_DIR, label),
        exist_ok=True
    )

os.makedirs(
    os.path.join(OUTPUT_DIR, "Uncertain"),
    exist_ok=True
)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading trained AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")
print("Classes:", class_names)

# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # MobileNetV2 preprocessing is already inside
    # the trained model.

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_class = class_names[predicted_index]

    confidence = float(
        prediction[predicted_index]
    )

    return predicted_class, confidence


# ------------------------------------------------------------
# Convert model class into broad condition label
# ------------------------------------------------------------

def get_condition(predicted_class):

    if predicted_class.lower().startswith("fresh"):
        return "Fresh"

    return "Spoilage"


# ------------------------------------------------------------
# Process all images
# ------------------------------------------------------------

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

rows = []

total_images = 0
high_confidence = 0
uncertain = 0

print("\nStarting automatic labelling...\n")

for root, dirs, files in os.walk(IMAGE_DIR):

    for filename in files:

        if not filename.lower().endswith(
            image_extensions
        ):
            continue

        image_path = os.path.join(
            root,
            filename
        )

        total_images += 1

        try:

            predicted_class, confidence = predict_image(
                image_path
            )

            confidence_percent = confidence * 100

            if confidence >= CONFIDENCE_THRESHOLD:

                condition = get_condition(
                    predicted_class
                )

                destination_folder = os.path.join(
                    OUTPUT_DIR,
                    condition
                )

                high_confidence += 1

            else:

                condition = "Uncertain"

                destination_folder = os.path.join(
                    OUTPUT_DIR,
                    "Uncertain"
                )

                uncertain += 1

            destination_path = os.path.join(
                destination_folder,
                filename
            )

            # Avoid duplicate filenames
            if os.path.exists(destination_path):

                base, ext = os.path.splitext(filename)

                counter = 1

                while os.path.exists(destination_path):

                    new_name = (
                        f"{base}_{counter}{ext}"
                    )

                    destination_path = os.path.join(
                        destination_folder,
                        new_name
                    )

                    counter += 1

            shutil.copy2(
                image_path,
                destination_path
            )

            rows.append([
                filename,
                image_path,
                predicted_class,
                condition,
                round(confidence_percent, 2)
            ])

            print(
                f"{total_images:4d} | "
                f"{filename[:35]:35} | "
                f"{predicted_class:20} | "
                f"{confidence_percent:6.2f}% | "
                f"{condition}"
            )

        except Exception as e:

            print(
                f"ERROR: {image_path}"
            )

            print(
                str(e)
            )


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "image_name",
        "original_path",
        "model_prediction",
        "condition_label",
        "confidence_percent"
    ])

    writer.writerows(rows)


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("\n======================================")
print(" AUTOMATIC LABELLING COMPLETE")
print("======================================")

print(
    f"Total images       : {total_images}"
)

print(
    f"High-confidence    : {high_confidence}"
)

print(
    f"Uncertain          : {uncertain}"
)

print(
    f"\nCSV saved at:"
)

print(
    CSV_PATH
)

print(
    f"\nLabelled dataset:"
)

print(
    OUTPUT_DIR
)

print("\nDone!")