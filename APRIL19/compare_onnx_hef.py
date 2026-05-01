#!/usr/bin/env python3
import argparse
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


def preprocess_onnx(frame, size=640):
    img = cv2.resize(frame, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)

    return x, img


def run_onnx(session, input_name, frame):
    x, vis = preprocess_onnx(frame)

    start = time.perf_counter()
    outputs = session.run(None, {input_name: x})
    end = time.perf_counter()

    return outputs, end - start, vis


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

    input_vstreams_params = InputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32
    )

    output_vstreams_params = OutputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32
    )

    return (
        target,
        network_group,
        network_group_params,
        input_vstreams_params,
        output_vstreams_params,
    )


def run_hef(network_group, network_group_params, input_params, output_params, frame):
    input_name = list(input_params.keys())[0]

    img = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hailo usually expects NHWC input
    x = np.expand_dims(rgb.astype(np.float32), axis=0)

    start = time.perf_counter()

    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            outputs = infer_pipeline.infer({input_name: x})

    end = time.perf_counter()

    return outputs, end - start, img


def flatten_output(output):
    if isinstance(output, dict):
        arrs = [np.array(v).flatten() for v in output.values()]
    elif isinstance(output, list):
        arrs = [np.array(v).flatten() for v in output]
    else:
        arrs = [np.array(output).flatten()]

    return np.concatenate(arrs)


def summarize_output(output):
    flat = flatten_output(output)

    return {
        "shape_total": flat.shape[0],
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "num_above_0.25": int(np.sum(flat > 0.25)),
        "num_above_0.50": int(np.sum(flat > 0.50)),
        "num_above_0.75": int(np.sum(flat > 0.75)),
    }


def compare_summaries(onnx_sum, hef_sum):
    return {
        "max_diff": abs(onnx_sum["max"] - hef_sum["max"]),
        "mean_diff": abs(onnx_sum["mean"] - hef_sum["mean"]),
        "std_diff": abs(onnx_sum["std"] - hef_sum["std"]),
        "above_0.50_diff": abs(
            onnx_sum["num_above_0.50"] - hef_sum["num_above_0.50"]
        ),
    }


def save_graphs(frame_count, onnx_times, hef_times, avg_onnx, avg_hef, agreement_diffs):
    frames = np.arange(1, frame_count + 1)

    plt.figure()
    plt.plot(frames, onnx_times * 1000, label="ONNX latency")
    plt.plot(frames, hef_times * 1000, label="HEF latency")
    plt.xlabel("Frame")
    plt.ylabel("Latency (ms)")
    plt.title("ONNX vs HEF Inference Latency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("onnx_vs_hef_latency.png", dpi=300)

    plt.figure()
    plt.bar(["ONNX", "HEF"], [1 / avg_onnx, 1 / avg_hef])
    plt.ylabel("FPS")
    plt.title("Average ONNX vs HEF FPS")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("onnx_vs_hef_fps.png", dpi=300)

    mean_diffs = [d["mean_diff"] for d in agreement_diffs]

    plt.figure()
    plt.plot(frames, mean_diffs, label="Mean output difference")
    plt.xlabel("Frame")
    plt.ylabel("Difference")
    plt.title("ONNX vs HEF Raw Output Difference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("onnx_vs_hef_output_difference.png", dpi=300)

    print("\nSaved graphs:")
    print("  onnx_vs_hef_latency.png")
    print("  onnx_vs_hef_fps.png")
    print("  onnx_vs_hef_output_difference.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="detector.onnx")
    parser.add_argument("--hef", default="detector.hef")
    parser.add_argument("--source", required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    print("[INFO] Loading ONNX model:", args.onnx)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("[INFO] ONNX input name:", input_name)

    print("[INFO] Loading HEF model:", args.hef)
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

    onnx_times = []
    hef_times = []
    agreement_diffs = []

    frame_count = 0

    print("\n[INFO] Starting comparison...\n")

    while frame_count < args.frames:
        ret, frame = cap.read()

        if not ret:
            break

        onnx_out, onnx_t, vis = run_onnx(session, input_name, frame)

        hef_out, hef_t, _ = run_hef(
            network_group,
            network_group_params,
            input_params,
            output_params,
            frame
        )

        onnx_sum = summarize_output(onnx_out)
        hef_sum = summarize_output(hef_out)
        diff = compare_summaries(onnx_sum, hef_sum)

        onnx_times.append(onnx_t)
        hef_times.append(hef_t)
        agreement_diffs.append(diff)

        frame_count += 1

        print(
            f"Frame {frame_count:04d} | "
            f"ONNX: {onnx_t * 1000:7.2f} ms | "
            f"HEF: {hef_t * 1000:7.2f} ms | "
            f"ONNX max: {onnx_sum['max']:.4f} | "
            f"HEF max: {hef_sum['max']:.4f} | "
            f"mean diff: {diff['mean_diff']:.6f}"
        )

        if args.show:
            cv2.imshow("Benchmark input", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if frame_count == 0:
        print("[ERROR] No frames were processed.")
        return

    onnx_times = np.array(onnx_times)
    hef_times = np.array(hef_times)

    avg_onnx = np.mean(onnx_times)
    avg_hef = np.mean(hef_times)

    print("\n========== RESULTS ==========")
    print(f"Frames tested: {frame_count}")

    print("\n--- Timing ---")
    print(f"ONNX avg latency: {avg_onnx * 1000:.2f} ms")
    print(f"HEF  avg latency: {avg_hef * 1000:.2f} ms")
    print(f"ONNX FPS: {1 / avg_onnx:.2f}")
    print(f"HEF  FPS: {1 / avg_hef:.2f}")
    print(f"Speedup HEF vs ONNX: {avg_onnx / avg_hef:.2f}x")

    print("\n--- Output similarity rough check ---")
    mean_max_diff = np.mean([d["max_diff"] for d in agreement_diffs])
    mean_mean_diff = np.mean([d["mean_diff"] for d in agreement_diffs])
    mean_std_diff = np.mean([d["std_diff"] for d in agreement_diffs])
    mean_conf_count_diff = np.mean([d["above_0.50_diff"] for d in agreement_diffs])

    print(f"Average max-output difference:       {mean_max_diff:.6f}")
    print(f"Average mean-output difference:      {mean_mean_diff:.6f}")
    print(f"Average std-output difference:       {mean_std_diff:.6f}")
    print(f"Average >0.50 activation difference: {mean_conf_count_diff:.2f}")

    save_graphs(
        frame_count,
        onnx_times,
        hef_times,
        avg_onnx,
        avg_hef,
        agreement_diffs
    )

    print("\n[NOTE]")
    print("This compares timing well, but the accuracy check is only approximate.")
    print("For true accuracy, you need labeled images and YOLO post-processing/NMS.")


if __name__ == "__main__":
    main()
  
    print(f"ONNX FPS: {1 / avg_onnx:.2f}")
    print(f"HEF  FPS: {1 / avg_hef:.2f}")

    print(f"Speedup HEF vs ONNX: {avg_onnx / avg_hef:.2f}x")

    print("\n--- Output similarity rough check ---")
    mean_max_diff = np.mean([d["max_diff"] for d in agreement_diffs])
    mean_mean_diff = np.mean([d["mean_diff"] for d in agreement_diffs])
    mean_std_diff = np.mean([d["std_diff"] for d in agreement_diffs])
    mean_conf_count_diff = np.mean([d["above_0.50_diff"] for d in agreement_diffs])

    print(f"Average max-output difference:       {mean_max_diff:.6f}")
    print(f"Average mean-output difference:      {mean_mean_diff:.6f}")
    print(f"Average std-output difference:       {mean_std_diff:.6f}")
    print(f"Average >0.50 activation difference: {mean_conf_count_diff:.2f}")

    save_graphs(
        frame_count,
        onnx_times,
        hef_times,
        avg_onnx,
        avg_hef,
        agreement_diffs
    )

    print("\n[NOTE]")
    print("This compares timing well, but the accuracy check is only approximate.")
    print("For true accuracy, you need labeled images and YOLO post-processing/NMS.")


if __name__ == "__main__":
    main()
