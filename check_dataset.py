import os

BASE_DIR = r"E:\FreshLife_AI\dataset"

print("\n===== DATASET CHECK =====\n")

for root, dirs, files in os.walk(BASE_DIR):
    image_files = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if image_files:
        print(f"{root}")
        print(f"Images: {len(image_files)}")
        print(f"Example: {image_files[:3]}")
        print("-" * 60)