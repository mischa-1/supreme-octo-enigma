# THIS FROM CHAT TO TEST YOLO

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

print("camera ret:", ret, "frame is None:", frame is None)
if not ret:
    raise SystemExit("Camera not returning frames")

results = model(frame, device="cpu", classes=[0], verbose=False)
annotated = results[0].plot()

cv2.imwrite("yolo_out.jpg", annotated)
print("wrote yolo_out.jpg")
