import os
import csv
import shutil
import cv2
import numpy as np

# ==============================
# PATHS
# ==============================

BASE_DIR = r"E:\FreshLife_AI"

SOURCE_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "auto_maturity_dataset")
CSV_PATH = os.path.join(BASE_DIR, "auto_maturity_labels.csv")

# ==============================
# SETTINGS
# ==============================

CONFIDENCE_THRESHOLD = 0.75

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================

for label in ["Unripe", "Ripe", "Overripe", "Uncertain"]:
    os.makedirs(
        os.path.join(OUTPUT_DIR, label),
        exist_ok=True
    )

# ==============================
# MATURITY ESTIMATION
# ==============================

def estimate_maturity(image_path, fruit_class):

    image = cv2.imread(image_path)

    if image is None:
        return "Uncertain", 0.0

    image = cv2.resize(image, (224, 224))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Remove very dark pixels
    mask = cv2.inRange(
        hsv,
        np.array([10, 30, 30]),
        np.array([179, 255, 255])
    )

    pixels = hsv[mask > 0]

    if len(pixels) < 100:
        return "Uncertain", 0.0

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]

    # ==============================
    # COLOR RATIOS
    # ==============================

    green = np.mean(
        ((h >= 35) & (h <= 85) & (s > 50))
    )

    yellow = np.mean(
        ((h >= 20) & (h < 35) & (s > 50))
    )

    orange = np.mean(
        ((h >= 8) & (h < 20) & (s > 60))
    )

    red = np.mean(
        ((h <= 8) | (h >= 170)) & (s > 60)
    )

    brown = np.mean(
        (h >= 5) &
        (h <= 25) &
        (s > 60) &
        (v < 150)
    )

    # ==============================
    # FRUIT-SPECIFIC RULES
    # ==============================

    fruit = fruit_class.lower()

    # --------------------------------
    # ORANGE
    # --------------------------------

    if "orange" in fruit:

        if green > 0.35:
            maturity = "Unripe"
            confidence = min(0.95, 0.70 + green * 0.5)

        elif orange > 0.30 or yellow > 0.30:
            maturity = "Ripe"
            confidence = min(
                0.95,
                0.70 + max(orange, yellow) * 0.5
            )

        elif brown > 0.25:
            maturity = "Overripe"
            confidence = min(0.95, 0.70 + brown * 0.5)

        else:
            return "Uncertain", 0.0

    # --------------------------------
    # MANGO
    # --------------------------------

    elif "mango" in fruit:

        if green > 0.40:
            maturity = "Unripe"
            confidence = min(0.92, 0.70 + green * 0.5)

        elif yellow > 0.25 or orange > 0.25:
            maturity = "Ripe"
            confidence = min(
                0.92,
                0.70 + max(yellow, orange) * 0.5
            )

        elif brown > 0.25:
            maturity = "Overripe"
            confidence = min(0.92, 0.70 + brown * 0.5)

        else:
            return "Uncertain", 0.0

    # --------------------------------
    # APPLE
    # --------------------------------

    elif "apple" in fruit:

        if green > 0.45:
            maturity = "Unripe"
            confidence = min(0.90, 0.70 + green * 0.4)

        elif red > 0.30 or yellow > 0.30:
            maturity = "Ripe"
            confidence = min(
                0.90,
                0.70 + max(red, yellow) * 0.4
            )

        elif brown > 0.30:
            maturity = "Overripe"
            confidence = min(0.90, 0.70 + brown * 0.4)

        else:
            return "Uncertain", 0.0

    # --------------------------------
    # STRAWBERRY
    # --------------------------------

    elif "strawberry" in fruit:

        if green > 0.30:
            maturity = "Unripe"
            confidence = min(0.90, 0.70 + green * 0.4)

        elif red > 0.35:
            maturity = "Ripe"
            confidence = min(0.90, 0.70 + red * 0.4)

        elif brown > 0.30:
            maturity = "Overripe"
            confidence = min(0.90, 0.70 + brown * 0.4)

        else:
            return "Uncertain", 0.0

    # --------------------------------
    # UNKNOWN FRUIT
    # --------------------------------

    else:
        return "Uncertain", 0.0

    # ==============================
    # CONFIDENCE FILTER
    # ==============================

    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain", confidence

    return maturity, confidence


# ==============================
# PROCESS DATASET
# ==============================

total = 0
unripe_count = 0
ripe_count = 0
overripe_count = 0
uncertain_count = 0

csv_rows = []

print("=" * 50)
print("AUTOMATIC MATURITY LABELLING")
print("=" * 50)

for fruit_class in os.listdir(SOURCE_DIR):

    class_path = os.path.join(
        SOURCE_DIR,
        fruit_class
    )

    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing: {fruit_class}")

    for filename in os.listdir(class_path):

        if not filename.lower().endswith(
            IMAGE_EXTENSIONS
        ):
            continue

        total += 1

        source_path = os.path.join(
            class_path,
            filename
        )

        label, confidence = estimate_maturity(
            source_path,
            fruit_class
        )

        confidence_percent = round(
            confidence * 100,
            2
        )

        destination_folder = os.path.join(
            OUTPUT_DIR,
            label
        )

        destination_path = os.path.join(
            destination_folder,
            f"{fruit_class}_{filename}"
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        csv_rows.append([
            filename,
            fruit_class,
            label,
            confidence_percent
        ])

        if label == "Unripe":
            unripe_count += 1

        elif label == "Ripe":
            ripe_count += 1

        elif label == "Overripe":
            overripe_count += 1

        else:
            uncertain_count += 1


# ==============================
# SAVE CSV
# ==============================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "original_class",
        "maturity_label",
        "confidence"
    ])

    writer.writerows(csv_rows)


# ==============================
# FINAL REPORT
# ==============================

print("\n")
print("=" * 50)
print("AUTOMATIC LABELLING COMPLETE")
print("=" * 50)

print(f"Total images       : {total}")
print(f"Unripe             : {unripe_count}")
print(f"Ripe               : {ripe_count}")
print(f"Overripe           : {overripe_count}")
print(f"Uncertain          : {uncertain_count}")

print("\nConfidence threshold:")
print(f"{CONFIDENCE_THRESHOLD * 100:.0f}%")

print("\nCSV saved at:")
print(CSV_PATH)

print("\nDataset saved at:")
print(OUTPUT_DIR)

print("\nDone!")