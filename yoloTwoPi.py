import cv2
import torch
from ultralytics import YOLO
import platform
from picamera2 import Picamera2
import time


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


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    #print(f"Using device: {device}")

    model = YOLO("yolov8n.pt")
    model.to(device)

    #print("Debug 1")
    #print("Debug 1", flush=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera error (cap.isOpened() is False)", flush=True)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            frame_rgb = picam2.capture_array()     # RGB numpy array
            frame = frame_rgb[:, :, ::-1].copy()   # RGB → BGR for OpenCV
            h, w, c = frame_rgb.shape

            # ---- YOLO + your existing logic ----
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                verbose=False
            )

            annotated_frame = frame.copy()
            annotated_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


            if results and results[0].boxes is not None:
                #print("Results if statement")
                boxes = results[0].boxes.xyxy.cpu().numpy()

                if len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    cv2.rectangle(
                        annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.circle(
                        annotated_frame, (cx, cy), 6, (0, 0, 255), -1
                    )

                    left_bound = w / 3
                    right_bound = 2 * w / 3

                    if cx > right_bound:
                        zone = "RIGHT"
                        # SEND TO OTHER PI GPIO 25

                    elif cx < left_bound:
                        zone = "LEFT"
                        # SEND TO OTHER PI GPIO 23

                    else:
                        zone = "MIDDLE"
                        # SEND TO OTHER PI GPIO 25


                    #print(zone)

                    cv2.putText(
                        annotated_frame,
                        f"ZONE: {zone}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

            cv2.imshow("Person Direction Feedback", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Ensure we always shut things down cleanly
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
