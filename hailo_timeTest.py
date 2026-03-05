import time
import numpy as np
import argparse
from pathlib import Path

from picamera2 import Picamera2
import cv2

import hailo_platform as hpf


N_DEFAULT = 100


def main():
    parser = argparse.ArgumentParser(description="Timing test: Picamera2 -> Hailo AI HAT+ (.hef)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model (.pt or .onnx)"
    )
    parser.add_argument("--output", default="hailo_time_output", type=str, help="CSV output name (with/without .csv)")
    parser.add_argument("--cam_w", type=int, default=640, help="Camera width (pixels)")
    parser.add_argument("--cam_h", type=int, default=480, help="Camera height (pixels)")
    parser.add_argument("--n", type=int, default=100, help="Number of frames to run inference on")
    args = parser.parse_args()

    # Output file handling (same style you like)
    filename = args.output
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
    output_dir = Path("time_data")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / filename

    # ---- Load HEF + configure Hailo device ----
    model_path = Path("models") / args.model
    hef = hpf.HEF(str(model_path))

    with hpf.VDevice() as vdevice:
        cfg = hpf.ConfigureParams.create_from_hef(
            hef, interface=hpf.HailoStreamInterface.PCIe
        )
        network_group = vdevice.configure(hef, cfg)[0]
        ng_params = network_group.create_params()

        # Stream info
        in_info = hef.get_input_vstream_infos()[0]
        out_infos = hef.get_output_vstream_infos()

        # Start with FLOAT32 streams (easiest). If you want max speed later,
        # switch to quantized=True and handle dequantization.
        in_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=True, format_type=hpf.FormatType.UINT8
        )
        out_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=True, format_type=hpf.FormatType.UINT8
        )

        in_h, in_w, in_c = in_info.shape  # typically H,W,C
        #print(f"Using HEF: {args.hef}")
        print(f"HEF input: {in_info.name} shape={in_info.shape}")
        print("HEF outputs:")
        for oi in out_infos:
            print(f"  - {oi.name} shape={oi.shape}")

        # ---- Camera setup ----
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (args.cam_w, args.cam_h), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.2)

        times_ms = np.empty(args.n, dtype=np.float64)

        try:
            with network_group.activate(ng_params):
                with hpf.InferVStreams(network_group, in_params, out_params) as pipe:
                    for i in range(args.n):
                        frame_rgb = picam2.capture_array()  # RGB uint8

                        # Resize to HEF input
                        resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

                        # Convert to float32. Many pipelines use [0,1]. If your results look wrong later,
                        # try removing "/ 255.0" (some HEFs expect 0..255 float).
                        input_data = {in_info.name: np.expand_dims(resized, axis=0)}  # uint8

                        t0 = time.perf_counter()
                        outputs = pipe.infer(input_data)
                        t1 = time.perf_counter()

                        times_ms[i] = (t1 - t0) * 1000.0

                        # Print output structure once
                        if i == 0:
                            print("Output keys:", list(outputs.keys()))
                            for k, v in outputs.items():
                                arr = np.asarray(v)
                                print(f"  {k}: {arr.shape} {arr.dtype}")

        finally:
            picam2.stop()

    np.savetxt(file_path, times_ms, delimiter=",")
    print(f"Saved CSV to: {file_path}")
    print(f"Mean inference (ms):   {times_ms.mean():.2f}")
    print(f"Median inference (ms): {np.median(times_ms):.2f}")


if __name__ == "__main__":
    main()
