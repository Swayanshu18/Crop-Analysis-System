import os
import shutil
from sklearn.model_selection import train_test_split

# ------------------------------
# CONFIGURATION
# ------------------------------

# Original PlantVillage folder
DATASET_ROOT = r"D:\crop\data\disease_masks\images\PlantVillage"


# Output folder (for train/val/test)
OUTPUT_ROOT = r"D:\crop\backend\data\plant"

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10


# ------------------------------
# HELPERS
# ------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ------------------------------
# MAIN
# ------------------------------

def main():
    print("Reading classes...")
    classes = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]

    print(f"Found {len(classes)} classes.")
    print(classes)

    for c in classes:
        class_dir = os.path.join(DATASET_ROOT, c)
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                  if img.lower().endswith((".jpg", ".jpeg", ".png"))]

        if len(images) < 3:
            print(f"⚠ Skipping class {c}, not enough images.")
            continue

        # Split
        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - TRAIN_SPLIT), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        # Destinations
        for name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            out_dir = os.path.join(OUTPUT_ROOT, name, c)
            ensure_dir(out_dir)

            # Copy files
            for img_path in imgs:
                shutil.copy(img_path, os.path.join(out_dir, os.path.basename(img_path)))

        print(f"✔ Done: {c}")

    print("\n🎉 Dataset successfully prepared!")
    print(f"Output saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
