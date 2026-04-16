import cv2
from ultralytics import YOLO

MODEL_PATH = "/Users/natashaigic/runs/ped_signal_smallobj_tuned_640test_next13/weights/best.pt"
VIDEO_PATH = "/Users/natashaigic/PycharmProjects/supreme-octo-enigma/ultralytics/IMG_6473.MOV"
OUTPUT_PATH = "output_video_testvid.mp4"

CONF_THRESHOLD = 0.25  #used ot be .5?

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print(model.names)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # IMPORTANT: no filtering here
    results = model(frame, conf=0.001)

    boxes = results[0].boxes

    signal_state = "UNKNOWN"

    best_state = None
    best_conf = 0

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            label = model.names[cls_id]

            # pick MOST confident state (fixes overwrite bug)
            if conf > best_conf:
                best_conf = conf

                if label == "ped_walk":
                    best_state = "WALK"
                elif label == "ped_stop":
                    best_state = "STOP"
                elif label == "ped_crosswalk":
                    best_state = "CROSSWALK"

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    if best_state is None:
        best_state = "UNKNOWN"

    cv2.putText(
        frame,
        f"Signal: {best_state}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )

    out.write(frame)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done. Saved to {OUTPUT_PATH}")