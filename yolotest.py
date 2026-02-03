import cv2
import torch
from ultralytics import YOLO
import platform

IS_PI = platform.system() == "Linux"

# Use BCM pin numbers (same numbers you used with GPIO.BCM)
MOTOR_RIGHT = 17   # Vibration motor 1
MOTOR_MIDDLE = 27  # Vibration motor 2
MOTOR_LEFT = 22    # Vibration motor 3

# lgpio globals (only valid on Pi)
_lgpio = None
_chip_handle = None

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
    print(f"Using device: {device}")

    model = YOLO("yolov8n.pt")
    model.to(device)

    cap = cv2.VideoCapture(0)
    #gst = (
    #    "libcamerasrc ! "
    #    "video/x-raw,format=I420,width=640,height=480,framerate=30/1 ! "
    #    "videoconvert ! appsink drop=true sync=false"
    #)
    #cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Camera error")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            results = model.track(
                frame,
                persist=True,
                classes=[0],  # Only person tracking
                verbose=False
            )

            annotated_frame = frame.copy()
            motors_off()

            if results and results[0].boxes is not None:
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


                    print(zone)

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
