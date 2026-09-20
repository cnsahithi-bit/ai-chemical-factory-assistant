from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import os

# =========================
# MODEL
# =========================
MODEL_PATH = r"runs\classify\train\weights\best.pt"

# =========================
# TRUE TEST SET
# =========================
TEST_DIR = r"dataset_small_final\test"

CLASSES = [
    "crack",
    "hole",
    "normal",
    "rust",
    "scratch"
]

IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)

print("Loading model...")
model = YOLO(MODEL_PATH)

y_true = []
y_pred = []
top5_correct = 0

print("\nEvaluating TEST SET...\n")

for true_class in CLASSES:

    folder = os.path.join(TEST_DIR, true_class)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"{true_class}: {len(images)} images")

    for image in images:

        image_path = os.path.join(folder, image)

        result = model.predict(
            source=image_path,
            verbose=False
        )[0]

        # Top-1
        predicted_index = result.probs.top1
        predicted_class = result.names[predicted_index]

        # Top-5
        top5_indices = result.probs.top5
        top5_classes = [
            result.names[i] for i in top5_indices
        ]

        y_true.append(true_class)
        y_pred.append(predicted_class)

        if true_class in top5_classes:
            top5_correct += 1


# =========================
# METRICS
# =========================

total = len(y_true)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    labels=CLASSES,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    labels=CLASSES,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASSES,
    average="weighted",
    zero_division=0
)

top5_accuracy = top5_correct / total


# =========================
# FINAL RESULTS
# =========================

print("\n")
print("=" * 60)
print("FINAL TEST SET RESULTS")
print("=" * 60)

print(f"Total test images   : {total}")
print(f"Top-1 Accuracy      : {accuracy * 100:.2f}%")
print(f"Top-5 Accuracy      : {top5_accuracy * 100:.2f}%")
print(f"Precision           : {precision * 100:.2f}%")
print(f"Recall              : {recall * 100:.2f}%")
print(f"F1 Score            : {f1 * 100:.2f}%")


# =========================
# PER CLASS
# =========================

print("\n")
print("=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASSES
)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(f"{'':12}", end="")

for cls in CLASSES:
    print(f"{cls:>10}", end="")

print()

for i, cls in enumerate(CLASSES):

    print(f"{cls:12}", end="")

    for j in range(len(CLASSES)):
        print(f"{cm[i][j]:>10}", end="")

    print()


print("\nEvaluation completed.")