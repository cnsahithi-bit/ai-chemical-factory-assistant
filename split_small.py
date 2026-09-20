import os
import random
import shutil

SOURCE = "dataset"
DEST = "dataset_small_final"

CLASSES = ["crack", "hole", "normal", "rust", "scratch"]

TOTAL_PER_CLASS = 500
TRAIN_COUNT = 350
VAL_COUNT = 100
TEST_COUNT = 50

random.seed(42)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST, split, cls), exist_ok=True)

for cls in CLASSES:

    source_folder = os.path.join(SOURCE, cls)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"\n{cls}: {len(images)} images found")

    if len(images) < TOTAL_PER_CLASS:
        raise ValueError(f"{cls} has fewer than 500 images!")

    random.shuffle(images)

    selected = images[:500]

    train_images = selected[:350]
    val_images = selected[350:450]
    test_images = selected[450:500]

    for image in train_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(DEST, "train", cls, image)
        )

    for image in val_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(DEST, "val", cls, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(DEST, "test", cls, image)
        )

    print(f"  Train: {len(train_images)}")
    print(f"  Val:   {len(val_images)}")
    print(f"  Test:  {len(test_images)}")

print("\n===================================")
print("DATASET SPLIT COMPLETED")
print("===================================")
print("Train: 1750 images")
print("Val:    500 images")
print("Test:   250 images")