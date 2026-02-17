import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="YOLO_PED/data.yaml",
    epochs=300,
    batch=8,
    imgsz=640,
    name="ped_signal_250img5"
)

save_dir = os.path.expanduser("~/runs/detect/ped_signal_250img5")
best_path = os.path.join(save_dir, "weights", "best.pt")

best_model = YOLO(best_path)
best_model.export(format="onnx")
