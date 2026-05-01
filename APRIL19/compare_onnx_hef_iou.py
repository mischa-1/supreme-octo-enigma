#!/usr/bin/env python3
import argparse
import csv
import time
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

try:
    from hailo_platform import (
        HEF,
        VDevice,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        InferVStreams,
        FormatType,
        HailoStreamInterface,
    )
    HAILO_AVAILABLE = True
except Exception:
    HAILO_AVAILABLE = False


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_onnx(frame, size=640):
    img = cv2.resize(frame, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # NHWC -> NCHW
    x = np.expand_dims(x, axis=0)
    return x, img


def preprocess_hef(frame, size=640):
    img = cv2.resize(frame, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hailo usually expects NHWC
    x = np.expand_dims(rgb.astype(np.float32), axis=0)
    return x, img


# -----------------------------
# Model loading
# -----------------------------
def load_onnx(onnx_path):
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    return session, input_name


def load_hef(hef_path):
    if not HAILO_AVAILABLE:
        raise RuntimeError("hailo_platform is not installed or not available.")

    hef = HEF(hef_path)
    target = VDevice()

    configure_params = ConfigureParams.create_from_hef(
        hef,
        interface=HailoStreamInterface.PCIe
    )

    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_params = InputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32
    )

    output_params = OutputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32
    )

    return target, network_group, network_group_params, input_params, output_params


# -----------------------------
# Inference
# -----------------------------
def run_onnx(session, input_name, frame):
    x, resized = preprocess_onnx(frame)

    start = time.perf_counter()
    outputs = session.run(None, {input_name: x})
    end = time.perf_counter()

    return outputs, end - start, resized


def run_hef(network_group, network_group_params, input_params, output_params, frame):
    input_name = list(input_params.keys())[0]
    x, resized = preprocess_hef(frame)

    start = time.perf_counter()

    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            outputs = infer_pipeline.infer({input_name: x})

    end = time.perf_counter()

    return outputs, end - start, resized


# -----------------------------
# Output handling
# -----------------------------
def output_to_array(output):
    """
    Converts ONNX list output or Hailo dict output into one numpy array.
    """
    if isinstance(output, dict):
        arrays = [np.array(v) for v in output.values()]
    elif isinstance(output, list):
        arrays = [np.array(v) for v in output]
    else:
        arrays = [np.array(output)]

    if len(arrays) == 1:
        return arrays[0]

    flattened = [a.reshape(-1, a.shape[-1]) if a.ndim >= 2 else a.reshape(-1, 1) for a in arrays]
    return np.concatenate(flattened, axis=0)


def normalize_yolo_shape(arr):
    """
    Tries to convert common YOLO output shapes into:
    [num_predictions, num_values]

    Handles:
    [1, 8400, 5]
    [1, 5, 8400]
    [8400, 5]
    [5, 8400]
    """
    arr = np.array(arr)

    arr = np.squeeze(arr)

    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])

    # If shape is [5, 8400] or [classes+4, predictions], transpose it
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 100:
        arr = arr.T

    return arr


def xywh_to_xyxy(x, y, w, h):
    return np.array([
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    ], dtype=np.float32)


def decode_yolo(output, conf_thresh=0.25, input_size=640):
    """
    Decodes YOLO-like output.

    Assumes rows are either:
    [x, y, w, h, confidence]
    or
    [x, y, w, h, objectness, class1, class2, ...]

    Returns:
    list of detections:
    {
      "box": [x1, y1, x2, y2],
      "score": confidence,
      "class_id": class_id
    }
    """
    arr = output_to_array(output)
    preds = normalize_yolo_shape(arr)

    detections = []

    if preds.shape[1] < 5:
        return detections

    boxes = preds[:, 0:4]

    # If coordinates look normalized, scale to pixels
    if np.nanmax(np.abs(boxes)) <= 2.0:
        boxes = boxes * input_size

    if preds.shape[1] == 5:
        scores = preds[:, 4]
        class_ids = np.zeros_like(scores, dtype=int)
    else:
        objectness = preds[:, 4]
        class_probs = preds[:, 5:]
        class_ids = np.argmax(class_probs, axis=1)
        class_scores = np.max(class_probs, axis=1)
        scores = objectness * class_scores

    for i in range(len(scores)):
        score = float(scores[i])

        if score < conf_thresh:
            continue

        x, y, w, h = boxes[i]
        box = xywh_to_xyxy(float(x), float(y), float(w), float(h))

        detections.append({
            "box": box,
            "score": score,
            "class_id": int(class_ids[i])
        })

    return detections


# -----------------------------
# NMS and IoU
# -----------------------------
def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def nms(detections, iou_thresh=0.45):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)

        remaining = []
        for det in detections:
            same_class = det["class_id"] == best["class_id"]
            overlap = compute_iou(best["box"], det["box"])

            if same_class and overlap > iou_thresh:
                continue

            remaining.append(det)

        detections = remaining

    return keep


def match_detections(onnx_dets, hef_dets, iou_match_thresh=0.5):
    """
    Matches ONNX detections to HEF detections using IoU.
    """
    matched_ious = []
    used_hef = set()

    for onnx_det in onnx_dets:
        best_iou = 0.0
        best_idx = None

        for j, hef_det in enumerate(hef_dets):
            if j in used_hef:
                continue

            if onnx_det["class_id"] != hef_det["class_id"]:
                continue

            iou = compute_iou(onnx_det["box"], hef_det["box"])

            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_idx is not None and best_iou >= iou_match_thresh:
            used_hef.add(best_idx)
            matched_ious.append(best_iou)

    return matched_ious


# -----------------------------
# Plotting
# -----------------------------
def save_report_figures(results):
    frames = np.array(results["frame"])

    onnx_latency = np.array(results["onnx_latency_ms"])
    hef_latency = np.array(results["hef_latency_ms"])
    mean_iou = np.array(results["mean_iou"])
    onnx_count = np.array(results["onnx_count"])
    hef_count = np.array(results["hef_count"])
    matches = np.array(results["matches"])

    avg_onnx_fps = 1000.0 / np.mean(onnx_latency)
    avg_hef_fps = 1000.0 / np.mean(hef_latency)

    # Figure 1: latency over frames
    plt.figure(figsize=(8, 5))
    plt.plot(frames, onnx_latency, linewidth=2, label="ONNX")
    plt.plot(frames, hef_latency, linewidth=2, label="HEF")
    plt.xlabel("Frame")
    plt.ylabel("Inference Latency (ms)")
    plt.title("ONNX vs HEF Inference Latency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_latency_over_time.png", dpi=300)

    # Figure 2: average FPS
    plt.figure(figsize=(6, 5))
    plt.bar(["ONNX", "HEF"], [avg_onnx_fps, avg_hef_fps])
    plt.ylabel("Average FPS")
    plt.title("Average Inference Speed")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_average_fps.png", dpi=300)

    # Figure 3: IoU agreement over frames
    plt.figure(figsize=(8, 5))
    plt.plot(frames, mean_iou, linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Mean Matched IoU")
    plt.title("Detection Agreement Between ONNX and HEF")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_iou_agreement.png", dpi=300)

    # Figure 4: detection counts
    plt.figure(figsize=(8, 5))
    plt.plot(frames, onnx_count, linewidth=2, label="ONNX detections")
    plt.plot(frames, hef_count, linewidth=2, label="HEF detections")
    plt.xlabel("Frame")
    plt.ylabel("Number of Detections")
    plt.title("Detection Count Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_detection_counts.png", dpi=300)

    # Figure 5: matched detections per frame
    plt.figure(figsize=(8, 5))
    plt.plot(frames, matches, linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Matched Detections")
    plt.title("Matched ONNX-HEF Detections per Frame")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_matched_detections.png", dpi=300)

    print("\nSaved report figures:")
    print("  figure_latency_over_time.png")
    print("  figure_average_fps.png")
    print("  figure_iou_agreement.png")
    print("  figure_detection_counts.png")
    print("  figure_matched_detections.png")


def save_csv(results, filename="onnx_hef_benchmark_results.csv"):
    keys = list(results.keys())

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(len(results[keys[0]])):
            writer.writerow([results[k][i] for k in keys])

    print(f"\nSaved CSV results: {filename}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx", default="detector.onnx")
    parser.add_argument("--hef", default="detector.hef")
    parser.add_argument("--source", required=True)
    parser.add_argument("--frames", type=int, default=100)

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.5)

    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    print("[INFO] Loading ONNX:", args.onnx)
    onnx_session, onnx_input_name = load_onnx(args.onnx)
    print("[INFO] ONNX input:", onnx_input_name)

    print("[INFO] Loading HEF:", args.hef)
    target, network_group, network_group_params, input_params, output_params = load_hef(args.hef)

    print("[INFO] HEF input streams:", list(input_params.keys()))
    print("[INFO] HEF output streams:", list(output_params.keys()))

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    results = {
        "frame": [],
        "onnx_latency_ms": [],
        "hef_latency_ms": [],
        "onnx_count": [],
        "hef_count": [],
        "matches": [],
        "mean_iou": [],
        "max_iou": [],
    }

    frame_idx = 0

    print("\n[INFO] Starting benchmark...\n")

    while frame_idx < args.frames:
        ret, frame = cap.read()

        if not ret:
            break

        onnx_output, onnx_time, vis = run_onnx(
            onnx_session,
            onnx_input_name,
            frame
        )

        hef_output, hef_time, _ = run_hef(
            network_group,
            network_group_params,
            input_params,
            output_params,
            frame
        )

        onnx_dets = decode_yolo(
            onnx_output,
            conf_thresh=args.conf,
            input_size=640
        )

        hef_dets = decode_yolo(
            hef_output,
            conf_thresh=args.conf,
            input_size=640
        )

        onnx_dets = nms(onnx_dets, iou_thresh=args.nms_iou)
        hef_dets = nms(hef_dets, iou_thresh=args.nms_iou)

        matched_ious = match_detections(
            onnx_dets,
            hef_dets,
            iou_match_thresh=args.match_iou
        )

        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        max_iou = float(np.max(matched_ious)) if matched_ious else 0.0

        frame_idx += 1

        results["frame"].append(frame_idx)
        results["onnx_latency_ms"].append(onnx_time * 1000)
        results["hef_latency_ms"].append(hef_time * 1000)
        results["onnx_count"].append(len(onnx_dets))
        results["hef_count"].append(len(hef_dets))
        results["matches"].append(len(matched_ious))
        results["mean_iou"].append(mean_iou)
        results["max_iou"].append(max_iou)

        print(
            f"Frame {frame_idx:04d} | "
            f"ONNX {onnx_time * 1000:7.2f} ms | "
            f"HEF {hef_time * 1000:7.2f} ms | "
            f"ONNX dets: {len(onnx_dets):2d} | "
            f"HEF dets: {len(hef_dets):2d} | "
            f"matches: {len(matched_ious):2d} | "
            f"mean IoU: {mean_iou:.3f}"
        )

        if args.show:
            display = vis.copy()

            for det in onnx_dets:
                x1, y1, x2, y2 = det["box"].astype(int)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

            cv2.imshow("ONNX detections shown in white", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if frame_idx == 0:
        print("[ERROR] No frames processed.")
        return

    onnx_latency = np.array(results["onnx_latency_ms"])
    hef_latency = np.array(results["hef_latency_ms"])
    mean_iou = np.array(results["mean_iou"])
    matches = np.array(results["matches"])

    print("\n========== SUMMARY ==========")
    print(f"Frames tested: {frame_idx}")

    print("\n--- Timing ---")
    print(f"ONNX avg latency: {np.mean(onnx_latency):.2f} ms")
    print(f"HEF  avg latency: {np.mean(hef_latency):.2f} ms")
    print(f"ONNX FPS: {1000.0 / np.mean(onnx_latency):.2f}")
    print(f"HEF  FPS: {1000.0 / np.mean(hef_latency):.2f}")
    print(f"HEF speedup: {np.mean(onnx_latency) / np.mean(hef_latency):.2f}x")

    print("\n--- Detection Agreement ---")
    print(f"Average matched detections/frame: {np.mean(matches):.2f}")
    print(f"Average mean IoU: {np.mean(mean_iou):.3f}")
    print(f"Frames with at least one match: {np.sum(matches > 0)} / {frame_idx}")

    save_csv(results)
    save_report_figures(results)

    print("\n[REPORT NOTE]")
    print("Use latency/FPS for speed comparison.")
    print("Use detection count and IoU for ONNX-vs-HEF output agreement.")
    print("If IoU is low, possible causes include quantization, different output scaling, or incorrect YOLO decoding assumptions.")


if __name__ == "__main__":
    main()
