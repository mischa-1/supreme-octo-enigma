import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
IMAGE_DIR = "/Users/natashaigic/Downloads/frame/labeled_frames/images"
LABEL_DIR = "/Users/natashaigic/Downloads/frame/labeled_frames/labels"
MODEL_PATH = "/Users/natashaigic/runs/ped_signal_smallobj_tuned_640test_next13/weights/best.pt"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.3

OUTPUT_LOG = "confusion_report.txt"

class_names = ["Wait", "Crosswalk", "Stop", "Walk"]
NUM_CLASSES = len(class_names)

model = YOLO(MODEL_PATH)

# =========================
# METRICS
# =========================
TP = np.zeros(NUM_CLASSES)
FP = np.zeros(NUM_CLASSES)
FN = np.zeros(NUM_CLASSES)

log_lines = []

# =========================
# IOU FUNCTION
# =========================
def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-6)

# =========================
# LOAD IMAGES
# =========================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print(f"Found {len(image_paths)} images")

# =========================
# MAIN LOOP
# =========================
for img_path in image_paths:

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    label_path = os.path.join(
        LABEL_DIR,
        os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    )

    # =====================
    # LOAD GT
    # =====================
    gt_boxes = []
    gt_classes = []

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                c = int(float(parts[0]))
                x, y, bw, bh = map(float, parts[1:])

                x1 = (x - bw / 2) * w
                y1 = (y - bh / 2) * h
                x2 = (x + bw / 2) * w
                y2 = (y + bh / 2) * h

                gt_boxes.append([x1, y1, x2, y2])
                gt_classes.append(c)

    # =====================
    # PREDICTIONS
    # =====================
    results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]

    pred_boxes = []
    pred_classes = []

    if results.boxes is not None and len(results.boxes) > 0:
        for b in results.boxes:
            pred_boxes.append(b.xyxy[0].cpu().numpy())
            pred_classes.append(int(b.cls[0].cpu().numpy()))

    # =====================
    # MATCHING
    # =====================
    matched_gt = set()

    for i, (p_box, p_cls) in enumerate(zip(pred_boxes, pred_classes)):

        best_iou = 0
        best_j = -1

        for j, g_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue

            iou = compute_iou(p_box, g_box)

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= IOU_THRESHOLD:
            matched_gt.add(best_j)

            g_cls = gt_classes[best_j]

            if g_cls == p_cls:
                TP[g_cls] += 1
            else:
                FP[p_cls] += 1
                FN[g_cls] += 1

                log_lines.append(
                    f"{os.path.basename(img_path)} | "
                    f"GT={class_names[g_cls]} -> PRED={class_names[p_cls]} | "
                    f"IOU={best_iou:.2f}"
                )
        else:
            FP[p_cls] += 1

            log_lines.append(
                f"{os.path.basename(img_path)} | "
                f"GT=NONE -> PRED={class_names[p_cls]} | "
                f"IOU={best_iou:.2f}"
            )

    # =====================
    # FALSE NEGATIVES
    # =====================
    for j, g_cls in enumerate(gt_classes):
        if j not in matched_gt:
            FN[g_cls] += 1

            log_lines.append(
                f"{os.path.basename(img_path)} | "
                f"GT={class_names[g_cls]} -> PRED=MISSING"
            )

# =========================
# SAVE LOG
# =========================
with open(OUTPUT_LOG, "w") as f:
    for line in log_lines:
        f.write(line + "\n")

# =========================
# OVERALL METRICS
# =========================
print("\n====================")
print("OVERALL RESULTS")
print("====================")

TP_total = np.sum(TP)
FP_total = np.sum(FP)
FN_total = np.sum(FN)

precision = TP_total / (TP_total + FP_total + 1e-6)
recall = TP_total / (TP_total + FN_total + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print(f"TP: {int(TP_total)}")
print(f"FP: {int(FP_total)}")
print(f"FN: {int(FN_total)}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# =========================
# PER CLASS RESULTS
# =========================
print("\n====================")
print("PER CLASS RESULTS")
print("====================")

for i in range(NUM_CLASSES):
    p = TP[i] / (TP[i] + FP[i] + 1e-6)
    r = TP[i] / (TP[i] + FN[i] + 1e-6)
    f = 2 * p * r / (p + r + 1e-6)

    print(f"\nClass: {class_names[i]}")
    print(f"TP: {int(TP[i])} FP: {int(FP[i])} FN: {int(FN[i])}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f:.4f}")

print(f"\nSaved confusion log to: {OUTPUT_LOG}")