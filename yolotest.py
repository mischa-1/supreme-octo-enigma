import cv2
import torch
from ultralytics import YOLO
import platform
from picamera2 import Picamera2
import time


IS_PI = platform.system() == "Linux"

# Use BCM pin numbers (same numbers you used with GPIO.BCM)
MOTOR_RIGHT = 17   # Vibration motor 1
MOTOR_MIDDLE = 27  # Vibration motor 2
MOTOR_LEFT = 22    # Vibration motor 3

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

if IS_PI:
    import lgpio as _lgpio

    # On Raspberry Pi, gpiochip0 is typically the main controller.
    # 0 here means /dev/gpiochip0
    _chip_handle = _lgpio.gpiochip_open(0)

    # Claim pins as outputs, initial state LOW (0)
    _lgpio.gpio_claim_output(_chip_handle, MOTOR_RIGHT, 0)
    _lgpio.gpio_claim_output(_chip_handle, MOTOR_MIDDLE, 0)
    _lgpio.gpio_claim_output(_chip_handle, MOTOR_LEFT, 0)

    def motors_off():
        _lgpio.gpio_write(_chip_handle, MOTOR_RIGHT, 0)
        _lgpio.gpio_write(_chip_handle, MOTOR_MIDDLE, 0)
        _lgpio.gpio_write(_chip_handle, MOTOR_LEFT, 0)

    def motor_on(pin: int):
        _lgpio.gpio_write(_chip_handle, pin, 1)

else:
    def motors_off():
        pass

    def motor_on(pin: int):
        pass


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    #print(f"Using device: {device}")

    model = YOLO("best.pt")
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
                        if IS_PI:
                            motor_on(MOTOR_RIGHT)

                    elif cx < left_bound:
                        zone = "LEFT"
                        if IS_PI:
                            motor_on(MOTOR_LEFT)

                    else:
                        zone = "MIDDLE"
                        if IS_PI:
                            motor_on(MOTOR_MIDDLE)


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

        if IS_PI:
            motors_off()
            # Release the gpiochip handle
            _lgpio.gpiochip_close(_chip_handle)


if __name__ == "__main__":
    main()
