from ultralytics import YOLO
import os


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = r"runs\classify\train\weights\best.pt"


# ============================================================
# TEST IMAGE
# ============================================================

TEST_DIR = r"dataset_small_final\test"

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# FIND FIRST IMAGE
# ============================================================

test_image = None

for root, dirs, files in os.walk(TEST_DIR):

    for file in files:

        if file.lower().endswith(IMAGE_EXTENSIONS):

            test_image = os.path.join(
                root,
                file
            )

            break

    if test_image is not None:
        break


if test_image is None:

    print("No test image found.")

    exit()


print("==============================")
print("YOLO TEST")
print("==============================")

print("Image:", test_image)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO model...")

model = YOLO(MODEL_PATH)


# ============================================================
# PREDICT
# ============================================================

result = model.predict(
    source=test_image,
    verbose=False
)[0]


# ============================================================
# GET TOP-1 PREDICTION
# ============================================================

predicted_index = result.probs.top1

predicted_class = result.names[
    predicted_index
]

confidence = float(
    result.probs.top1conf
)


# ============================================================
# OUTPUT
# ============================================================

print("\n==============================")
print("YOLO PREDICTION")
print("==============================")

print("Predicted class :", predicted_class)

print(
    "Confidence      :",
    round(confidence, 4)
)