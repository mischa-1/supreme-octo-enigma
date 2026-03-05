import cv2
import torch
from ultralytics import YOLO
import platform
from picamera2 import Picamera2
import time
import numpy as np
import csv
import argparse
from pathlib import Path

parse = argparse.ArgumentParser(description= "Date for this program")
parse.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to YOLO model (.pt or .onnx)"
)
parser.add_argument("--model2", required=True, help="Second model weights path (.pt/.onnx/etc)")
parse.add_argument("--output", action= "store", type=str, default="yolov8n.pt", help = "Name for CSV data file")
args = parse.parse_args()

IS_PI = platform.system() == "Linux"

# lgpio globals (only valid on Pi)
_lgpio = None
_chip_handle = None


N = 100


def main():
    
    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    dev = "cpu"

    is_exported = model_path_lower.endswith((".onnx", ".engine", ".tflite", ".openvino", ".mlpackage"))

    
    model1_path = Path("models") / args.model
    model2_path = Path("models") / args.model2

    print(f"Using model1: {model1_path}")
    print(f"Using model2: {model2_path}")

    model1 = YOLO(model1_path, task="detect")
    model2 = YOLO(model2_path, task="detect")


    if (not is_exported) and model_path_lower.endswith(".pt"):
        model.to(dev)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )

    picam2.configure(config)
    from libcamera import controls
    picam2.set_controls({"AfMode": controls.AfModeEnum.Manual})
    picam2.start()
    time.sleep(0.2)   # let camera warm up
    #print("Debug 1")
    #print("Debug 1", flush=True)

    times1 = []
    times2 = []
    times_total = []

    # cap = cv2.VideoCapture(0)
    #gst = (
    #    "libcamerasrc ! "
    #    "video/x-raw,format=I420,width=640,height=480,framerate=30/1 ! "
    #    "videoconvert ! appsink drop=true sync=false"
    #)
    #cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    #print("Debug 2")
    #print("Debug 2 - cap.isOpened():", cap.isOpened(), flush=True)

    #if not cap.isOpened():
    #    print("Camera error (cap.isOpened() is False)", flush=True)
    #    return

    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        for _ in range(3):
            _ = model1.predict(source=frame, verbose=False)
            _ = model2.predict(source=frame, verbose=False)
        for i in range(N):
            
            frame_rgb = picam2.capture_array()     # RGB numpy array
            frame = frame_rgb[:, :, ::-1].copy()   # RGB → BGR for OpenCV
            #h, w, c = frame_rgb.shape
            t0 = time.perf_counter()
            _ = model1.predict(source=frame, verbose=False)
            t1 = time.perf_counter()
            _ = model2.predict(source=frame, verbose=False)
            t2 = time.perf_counter()

            dt1 = (t1 - t0)
            dt2 = (t2 - t1)
            dtt = (t2 - t0)

            times1.append(dt1)
            times2.append(dt2)
            times_total.append(dtt)

            #annotated_frame = frame.copy()
            #annotated_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #motors_off()
            #ret, frame = cap.read()
            #print("read ret:", ret, "frame_none:", frame is None, flush=True)
            #if not ret:
            #    break
            #
            #h, w, _ = frame.shape

            #results = model.track(
            #    frame,
            #    persist=True,
            #    classes=[0],  # Only person tracking
            #    verbose=False
            #)

            #print("Debug before annotated_frame")
            #annotated_frame = frame.copy()
            #motors_off()

    finally:
        picam2.stop()
        # Ensure we always shut things down cleanly
        #cap.release()
    import os
  
    output_dir = Path("time_data")
    output_dir.mkdir(exist_ok=True)
  
    out1 = os.path.join(OUTPUT_DIR, args.output + "_model1.csv")
    out2 = os.path.join(OUTPUT_DIR, args.output + "_model2.csv")
    outt = os.path.join(OUTPUT_DIR, args.output + "_total.csv")

    with open(out1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "seconds"])
        for i, dt in enumerate(times1):
            w.writerow([i, dt])

    with open(out2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "seconds"])
        for i, dt in enumerate(times2):
            w.writerow([i, dt])

    with open(outt, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "seconds"])
        for i, dt in enumerate(times_total):
            w.writerow([i, dt])

    print("Wrote:", out1, out2, outt)



if __name__ == "__main__":
    main()
