import cv2
import torch
from ultralytics import YOLO
import platform
from picamera2 import Picamera2
import time
import numpy as np
import csv
import argparse

parse = argparse.ArgumentParser(description= "Date for this program")
parse.add_argument("--model", action= "store", type=str, default="Error: YOLO mdoel must be specified", help = "time for loop")
parse.add_argument("--output", action= "store", type=str, default="yolov8n.pt", help = "Name for CSV data file")
args = parse.parse_args()

IS_PI = platform.system() == "Linux"

# lgpio globals (only valid on Pi)
_lgpio = None
_chip_handle = None


N = 100


def main():
    
    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    #print(f"Using device: {device}")

    print("Using model:", args.model)

    model = YOLO(args.model)
    model.to(device)

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
    times = np.empty(N, dtype=np.float64)
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
        for i in range(N):
            
            frame_rgb = picam2.capture_array()     # RGB numpy array
            frame = frame_rgb[:, :, ::-1].copy()   # RGB → BGR for OpenCV
            #h, w, c = frame_rgb.shape

            startTime = time.time()

            # ---- YOLO + your existing logic ----
            results = model.predict(
                frame,
                verbose=False
            )

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
            times[i] = time.time() - startTime;

    finally:
        picam2.stop()
        # Ensure we always shut things down cleanly
        #cap.release()

    filename = args.output
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
    
    np.savetxt(filename, times, delimiter=',')


if __name__ == "__main__":
    main()
