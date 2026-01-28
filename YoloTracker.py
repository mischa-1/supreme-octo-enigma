import cv2
import torch
from ultralytics import YOLO
import platform

IS_PI = platform.system() == "Linux"

if IS_PI:
    import RPi.GPIO as GPIO

    GPIO.setmode(GPIO.BCM)

    MOTOR_RIGHT = 17   # Vibration motor 1
    MOTOR_MIDDLE = 27  # Vibration motor 2
    MOTOR_LEFT = 22    # Vibration motor 3

    GPIO.setup(MOTOR_RIGHT, GPIO.OUT)
    GPIO.setup(MOTOR_MIDDLE, GPIO.OUT)
    GPIO.setup(MOTOR_LEFT, GPIO.OUT)

    def motors_off():
        GPIO.output(MOTOR_RIGHT, GPIO.LOW)
        GPIO.output(MOTOR_MIDDLE, GPIO.LOW)
        GPIO.output(MOTOR_LEFT, GPIO.LOW)

else:

    def motors_off():
        pass


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolov8n.pt")
    model.to(device)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera error")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        results = model.track(
            frame,
            persist=True,
            classes=[0], # Only person tracking
            verbose=False
        )

        annotated_frame = frame.copy()
        motors_off()

        if results[0].boxes is not None:
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
                        GPIO.output(MOTOR_RIGHT, GPIO.HIGH)

                elif cx < left_bound:
                    zone = "LEFT"
                    if IS_PI:
                        GPIO.output(MOTOR_LEFT, GPIO.HIGH)

                else:
                    zone = "MIDDLE"
                    if IS_PI:
                        GPIO.output(MOTOR_MIDDLE, GPIO.HIGH)

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

    cap.release()
    cv2.destroyAllWindows()

    if IS_PI:
        motors_off()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
