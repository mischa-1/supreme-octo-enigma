import cv2
import torch
from ultralytics import YOLO
import platform
from picamera2 import Picamera2
import time
import numpy as py
import csv

IS_PI = platform.system() == "Linux"

# lgpio globals (only valid on Pi)
_lgpio = None
_chip_handle = None

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(0.2)   # let camera warm up

times = np.empty(100)
i = 0




def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    #print(f"Using device: {device}")

    model = YOLO("yolov8n.pt")
    model.to(device)

    #print("Debug 1")
    #print("Debug 1", flush=True)
    cap = cv2.VideoCapture(0)
    #gst = (
    #    "libcamerasrc ! "
    #    "video/x-raw,format=I420,width=640,height=480,framerate=30/1 ! "
    #    "videoconvert ! appsink drop=true sync=false"
    #)
    #cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    #print("Debug 2")
    #print("Debug 2 - cap.isOpened():", cap.isOpened(), flush=True)

    if not cap.isOpened():
        print("Camera error (cap.isOpened() is False)", flush=True)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while i < 100:
            
            frame_rgb = picam2.capture_array()     # RGB numpy array
            frame = frame_rgb[:, :, ::-1].copy()   # RGB → BGR for OpenCV
            h, w, c = frame_rgb.shape

            startTime = time.time()

            # ---- YOLO + your existing logic ----
            results = model.track(
                frame,
                persist=True,
                classes=[0],
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
            
            i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Ensure we always shut things down cleanly
        cap.release()

  [np.savetxt](https://numpy.org)('time_output.csv', times, delimiter=',')


if __name__ == "__main__":
    main()
