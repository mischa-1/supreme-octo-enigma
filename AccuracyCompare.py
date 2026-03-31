
import cv2
import csv
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# -------- CONFIG --------
MODEL_1_PATH = "/Users/natashaigic/runs/ped_signal_smallobj_tuned_640test_next13/weights/best.pt"
MODEL_2_PATH = "/Users/natashaigic/PycharmProjects/supreme-octo-enigma/ultralytics/runs/detect/train10/weights/best.pt"

IMAGES_DIR = "/Volumes/PhotoDrive/experiment/proj4/images"
LABELS_DIR = "/Volumes/PhotoDrive/experiment/proj4/labels"

OUTPUT_DIR = "comparison_output"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

SAVE_VIS = True  # set False if you don’t want annotated images
# ------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def xywh_norm_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xywh = list(map(float, parts[1:5]))
            xyxy = xywh_norm_to_xyxy(xywh, img_w, img_h)
            boxes.append((cls, xyxy))
    return boxes


def match_detections(gt_boxes, detections, iou_thresh):
    matched_gt = set()
    tp = fp = 0

    detections = sorted(detections, key=lambda x: -x[1])

    for cls_d, conf, box_d in detections:
        best_iou, best_idx = 0.0, -1

        for idx, (cls_g, box_g) in enumerate(gt_boxes):
            if idx in matched_gt or cls_g != cls_d:
                continue

            val = iou(box_d, box_g)
            if val > best_iou:
                best_iou, best_idx = val, idx

        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def draw_boxes(frame, detections, gt_boxes, model_names, color_det):
    vis = frame.copy()

    # Ground truth (white)
    for cls_g, box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Predictions
    for cls_d, conf, box in detections:
        x1, y1, x2, y2 = map(int, box)
        label = model_names.get(cls_d, str(cls_d))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_det, 2)
        cv2.putText(vis, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_det, 1)

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    vis_dir = out_dir / "visualizations"
    if SAVE_VIS:
        vis_dir.mkdir(exist_ok=True)

    print("Loading models...")
    model1 = YOLO(MODEL_1_PATH)
    model2 = YOLO(MODEL_2_PATH)

    image_paths = sorted(
        list(Path(IMAGES_DIR).glob("*.jpg")) +
        list(Path(IMAGES_DIR).glob("*.png"))
    )

    if not image_paths:
        raise RuntimeError("No images found.")

    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)

    writer.writerow([
        "image",
        "gt_boxes",
        "m1_tp", "m1_fp", "m1_fn", "m1_precision", "m1_recall", "m1_f1", "m1_time_ms",
        "m2_tp", "m2_fp", "m2_fn", "m2_precision", "m2_recall", "m2_f1", "m2_time_ms",
    ])

    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]

        label_file = Path(LABELS_DIR) / (img_path.stem + ".txt")
        gt_boxes = load_ground_truth(label_file, w, h)

        row = [img_path.name, len(gt_boxes)]

        vis_outputs = []

        for tag, model, color in [
            ("m1", model1, (0, 255, 0)),
            ("m2", model2, (255, 100, 0)),
        ]:
            import time
            t0 = time.perf_counter()
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            elapsed = (time.perf_counter() - t0) * 1000.0

            detections = []
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    detections.append((
                        int(box.cls[0]),
                        float(box.conf[0]),
                        box.xyxy[0].tolist()
                    ))

            tp, fp, fn = match_detections(gt_boxes, detections, IOU_THRESHOLD)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            row += [tp, fp, fn, round(precision, 4),
                    round(recall, 4), round(f1, 4), round(elapsed, 3)]

            if SAVE_VIS:
                vis = draw_boxes(frame, detections, gt_boxes, model.names, color)
                cv2.putText(vis, f"{tag.upper()} F1:{f1:.2f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                vis_outputs.append(vis)

        writer.writerow(row)

        # Save visualization (side-by-side image)
        if SAVE_VIS and len(vis_outputs) == 2:
            combined = np.hstack(vis_outputs)
            out_path = vis_dir / img_path.name
            cv2.imwrite(str(out_path), combined)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(image_paths)}")

    csv_file.close()

    print("\nDone!")
    print(f"CSV saved to: {csv_path}")
    if SAVE_VIS:
        print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()


#plot confusion matrix
# ===== CONFUSION MATRIX + METRICS VISUALIZATION =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Load your results CSV
df = pd.read_csv("/Users/natashaigic/PycharmProjects/supreme-octo-enigma/comparison_output/metrics.csv")  # <-- change filename if needed

# 🔹 Sum totals across all images
m1_tp = df["m1_tp"].sum()
m1_fp = df["m1_fp"].sum()
m1_fn = df["m1_fn"].sum()

m2_tp = df["m2_tp"].sum()
m2_fp = df["m2_fp"].sum()
m2_fn = df["m2_fn"].sum()

# 🔹 Create confusion-style matrices
m1_matrix = np.array([[m1_tp, m1_fn],
                      [m1_fp, 0]])

m2_matrix = np.array([[m2_tp, m2_fn],
                      [m2_fp, 0]])

# 🔹 Plot function
def plot_matrix(matrix, title):
    fig, ax = plt.subplots()
    ax.imshow(matrix)

    labels = [["TP", "FN"], ["FP", ""]]

    for i in range(2):
        for j in range(2):
            text = f"{labels[i][j]}\n{matrix[i, j]}"
            ax.text(j, i, text, ha="center", va="center")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Predicted Object", "No Prediction"])
    ax.set_yticklabels(["Actual Object", "No Object"])

    ax.set_title(title)

    plt.show()

# 🔹 Plot confusion matrices
plot_matrix(m1_matrix, "Model 1 Confusion Matrix")
plot_matrix(m2_matrix, "Model 2 Confusion Matrix")

# 🔹 Bar chart comparison
labels = ["TP", "FP", "FN"]
m1_values = [m1_tp, m1_fp, m1_fn]
m2_values = [m2_tp, m2_fp, m2_fn]

x = np.arange(len(labels))
width = 0.35

plt.figure()
plt.bar(x - width/2, m1_values, width, label="Model 1")
plt.bar(x + width/2, m2_values, width, label="Model 2")

plt.xticks(x, labels)
plt.title("Model Comparison: TP vs FP vs FN")
plt.legend()

plt.show()

# 🔹 Print summary stats (optional but useful)
print("\n=== TOTALS ===")
print(f"Model 1 -> TP: {m1_tp}, FP: {m1_fp}, FN: {m1_fn}")
print(f"Model 2 -> TP: {m2_tp}, FP: {m2_fp}, FN: {m2_fn}")

print("\n=== PRECISION / RECALL ===")
m1_precision = m1_tp / (m1_tp + m1_fp + 1e-6)
m1_recall = m1_tp / (m1_tp + m1_fn + 1e-6)

m2_precision = m2_tp / (m2_tp + m2_fp + 1e-6)
m2_recall = m2_tp / (m2_tp + m2_fn + 1e-6)

print(f"Model 1 -> Precision: {m1_precision:.3f}, Recall: {m1_recall:.3f}")
print(f"Model 2 -> Precision: {m2_precision:.3f}, Recall: {m2_recall:.3f}")