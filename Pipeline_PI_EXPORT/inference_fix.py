#!/usr/bin/env python3
"""
inference.py

Pedestrian Signal Inference
Stage 1: Detector
    - picamera mode  -> Hailo HEF detector (AI HAT)
    - webcam/video   -> ONNX detector by default
    - video/webcam can also be forced to HEF with --detector-backend hef
Stage 2: CNN+GRU classifier (.onnx via ONNX Runtime)

Examples:
  python3 inference.py --initialize
  python3 inference.py --mode picamera
  python3 inference.py --mode webcam
  python3 inference.py --mode webcam --detector-backend hef
  python3 inference.py --mode video --source sample.mp4
  python3 inference.py --mode video --source sample.mp4 --detector-backend hef
  python3 inference.py --mode video --source sample.mp4 --audio tts
  python3 inference.py --mode video --source sample.mp4 --save
"""

from __future__ import annotations

import os
import cv2
import json
import time
import argparse
import threading
import struct
import tempfile
import subprocess
import sys
from pathlib import Path
from collections import deque

import numpy as np
import onnxruntime as ort

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


# ──────────────────────────────────────────────
# PATHS / CONFIG
# ──────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

DETECTOR_HEF = SCRIPT_DIR / "detector.hef"
DETECTOR_ONNX = SCRIPT_DIR / "detector.onnx"
CLASSIFIER_ONNX = SCRIPT_DIR / "runs" / "classifier" / "best.onnx"
CLASS_MAP_PATH = SCRIPT_DIR / "runs" / "classifier" / "class_map.json"
REQUIREMENTS_PATH = SCRIPT_DIR / "requirements.txt"

CONF_THRESH = 0.05
PADDING = 0.20
SEQUENCE_LENGTH = 30
IMG_SIZE = 128

DISPLAY_W = 960
DISPLAY_H = 540

DEFAULT_AUDIO_MODE = "tones"
DEFAULT_SAVE_PATH = "output.mp4"

DET_INPUT_W = 640
DET_INPUT_H = 640

# HEF detector appears to be single-class object localization
DETECTOR_CLASS_NAMES = ["signal"]
VALID_DETECTOR_CLASS_IDS = {0}

TONE_REPEAT_INTERVAL = {
    "walk": 1.5,
    "caution": 1.0,
    "stop": 2.0,
}

SAMPLE_RATE = 44100


# ──────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────

def get_torch_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


TORCH_DEVICE = get_torch_device()


# ──────────────────────────────────────────────
# INITIALIZATION
# ──────────────────────────────────────────────

def run_command(cmd: list[str], check: bool = True) -> int:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def initialize_environment(requirements_path: Path) -> None:
    requirements_path = requirements_path.resolve()

    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements.txt not found: {requirements_path}")

    print(f"Using requirements file: {requirements_path}")
    print()

    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], check=True)

    print()
    print("Python dependency installation finished.")
    print()
    print("Notes:")
    print("- If you are on Raspberry Pi and using picamera2, you may also need:")
    print("    sudo apt update")
    print("    sudo apt install -y python3-picamera2 libatlas-base-dev libportaudio2")
    print("- If you are using the Hailo AI HAT, make sure the Hailo runtime/software stack")
    print("  is installed separately according to your Hailo version.")
    print("- The script will not auto-install Hailo system packages for you.")
    print()
    print("Initialization complete.")


# ──────────────────────────────────────────────
# TONE GENERATION
# ──────────────────────────────────────────────

def generate_tone(freq, duration, volume=0.6, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    fade = int(sample_rate * 0.01)
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32) * volume
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return wave


def concat_tones(tones, gap_ms=80):
    gap = np.zeros(int(SAMPLE_RATE * gap_ms / 1000), dtype=np.float32)
    result = []
    for i, tone in enumerate(tones):
        result.append(tone)
        if i < len(tones) - 1:
            result.append(gap)
    return np.concatenate(result)


def build_tones():
    return {
        "walk": concat_tones([
            generate_tone(800, 0.12),
            generate_tone(800, 0.12)
        ], gap_ms=60),
        "caution": concat_tones([
            generate_tone(520, 0.09),
            generate_tone(520, 0.09),
            generate_tone(520, 0.09)
        ], gap_ms=40),
        "stop": generate_tone(280, 0.25),
    }


def play_tone_array(wave):
    try:
        import sounddevice as sd
        sd.play(wave, samplerate=SAMPLE_RATE)
        sd.wait()
        return
    except ImportError:
        pass

    pcm = (wave * 32767).astype(np.int16).tobytes()
    n, sr, ch, bps = len(pcm), SAMPLE_RATE, 1, 16

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, ch, sr,
                            sr * ch * bps // 8, ch * bps // 8, bps))
        f.write(b"data")
        f.write(struct.pack("<I", n))
        f.write(pcm)

    try:
        if os.path.exists("/usr/bin/afplay"):
            subprocess.run(["afplay", tmp], capture_output=True)
        else:
            subprocess.run(["aplay", tmp], capture_output=True)
    finally:
        os.unlink(tmp)


# ──────────────────────────────────────────────
# AUDIO MANAGER
# ──────────────────────────────────────────────

class AudioManager:
    def __init__(self, mode="tones"):
        self.mode = mode
        self.last_state = None
        self.last_played = 0.0
        self._lock = threading.Lock()
        self._playing = False
        self.tones = build_tones() if mode == "tones" else {}
        self.tts_engine = None

        if mode == "tts" and HAS_PYTTSX3:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", 185)
            except Exception:
                self.tts_engine = None

    def _play_async(self, wave):
        def _run():
            with self._lock:
                self._playing = True
            try:
                play_tone_array(wave)
            finally:
                with self._lock:
                    self._playing = False
        threading.Thread(target=_run, daemon=True).start()

    def _speak_async(self, text):
        def _run():
            if self.tts_engine:
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    return
                except Exception:
                    pass
            try:
                subprocess.Popen(["say", text])
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    def update(self, state, real_detection):
        if state is None:
            return

        now = time.time()
        state_changed = state != self.last_state
        interval = TONE_REPEAT_INTERVAL.get(state, 2.0)
        due_for_repeat = (now - self.last_played) >= interval

        if self.mode == "tones":
            if real_detection and (state_changed or due_for_repeat):
                with self._lock:
                    already_playing = self._playing
                if not already_playing:
                    self._play_async(self.tones[state])
                    self.last_played = now

        elif self.mode == "tts":
            if state_changed:
                self._speak_async(state)

        self.last_state = state


# ──────────────────────────────────────────────
# DETECTOR BACKENDS
# ──────────────────────────────────────────────

def letterbox_image(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_w, new_h = new_shape
    scale = min(new_w / w, new_h / h)

    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    pad_x = (new_w - resized_w) // 2
    pad_y = (new_h - resized_h) // 2
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized

    return canvas, scale, pad_x, pad_y


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.array([], dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def nms_numpy(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return []

    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break

        ious = box_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thresh]

    return keep


def postprocess_yolo_like(output, orig_w, orig_h, conf_thresh=0.30, iou_thresh=0.45):
    pred = np.array(output)

    if pred.ndim == 3:
        pred = pred[0]

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.shape[1] < 5:
        return []

    _, scale, pad_x, pad_y = letterbox_image(
        np.zeros((orig_h, orig_w, 3), dtype=np.uint8),
        (DET_INPUT_W, DET_INPUT_H)
    )

    boxes = []
    scores = []
    class_ids = []

    for row in pred:
        cx, cy, bw, bh = row[:4]
        cls_scores = row[4:]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])

        if score < conf_thresh:
            continue
        if cls_id not in VALID_DETECTOR_CLASS_IDS:
            continue

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(cls_id)

    if not boxes:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    keep = nms_numpy(boxes, scores, iou_thresh=iou_thresh)

    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i].astype(int)
        detections.append((x1, y1, x2, y2, float(scores[i]), int(class_ids[i])))

    return detections


def postprocess_hef_single_class(output, orig_w, orig_h, conf_thresh=0.30, iou_thresh=0.45):
    pred = np.array(output, dtype=np.float32)

    # Expected HEF shape from your debug run: (1, 1, 8400, 5)
    if pred.ndim == 4:
        pred = pred[0, 0]
    elif pred.ndim == 3:
        pred = pred[0]

    if pred.ndim != 2 or pred.shape[1] != 5:
        print("Unexpected HEF output shape:", pred.shape)
        return []

    _, scale, pad_x, pad_y = letterbox_image(
        np.zeros((orig_h, orig_w, 3), dtype=np.uint8),
        (DET_INPUT_W, DET_INPUT_H)
    )

    boxes = []
    scores = []

    for row in pred:
        cx, cy, bw, bh, score = row

        if score < conf_thresh:
            continue

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))

    if not boxes:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    keep = nms_numpy(boxes, scores, iou_thresh=iou_thresh)

    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i].astype(int)
        detections.append((x1, y1, x2, y2, float(scores[i]), 0))

    return detections


class BaseDetector:
    def infer(self, frame):
        raise NotImplementedError


class OnnxDetector(BaseDetector):
    def __init__(self, path: Path):
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"ONNX detector not found: {path}")

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        print(f"Loaded ONNX detector: {path}")
        print(f"ONNX input: {self.input_name}")
        print(f"ONNX outputs: {self.output_names}")

    def preprocess(self, frame):
        img, _, _, _ = letterbox_image(frame, (DET_INPUT_W, DET_INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]
        return img

    def infer(self, frame):
        inp = self.preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: inp})

        # Current ONNX path still assumes first output is the detection tensor.
        # This matches your original code behavior.
        return postprocess_yolo_like(outputs[0], frame.shape[1], frame.shape[0], CONF_THRESH)


class HefDetector(BaseDetector):
    def __init__(self, path: Path):
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"HEF detector not found: {path}")

        self.path = path

        try:
            from hailo_platform import HEF, VDevice, ConfigureParams, HailoStreamInterface
            import hailo_platform
        except ImportError as e:
            raise RuntimeError(
                "hailo_platform is not installed. Install/activate the Hailo runtime on the Pi first."
            ) from e

        self.hailo_platform = hailo_platform
        self.HEF = HEF
        self.VDevice = VDevice
        self.ConfigureParams = ConfigureParams
        self.HailoStreamInterface = HailoStreamInterface

        self.hef = HEF(str(path))
        self.target = VDevice()
        self.configure_params = ConfigureParams.create_from_hef(
            hef=self.hef,
            interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_infos = self.hef.get_output_vstream_infos()

        self.input_name = self.input_vstream_info.name
        self.output_names = [o.name for o in self.output_vstream_infos]

        print(f"Loaded HEF detector: {path}")
        print(f"HEF input: {self.input_name}")
        print(f"HEF outputs: {self.output_names}")

    def preprocess(self, frame):
        img, _, _, _ = letterbox_image(frame, (DET_INPUT_W, DET_INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        return img

    def infer(self, frame):
        from hailo_platform import InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams

        inp = self.preprocess(frame)

        input_params = InputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=True,
            format_type=FormatType.UINT8
        )
        output_params = OutputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=FormatType.FLOAT32
        )

        with self.network_group.activate(self.network_group_params):
            with InferVStreams(self.network_group, input_params, output_params) as infer_pipeline:
                results = infer_pipeline.infer({self.input_name: np.expand_dims(inp, axis=0)})

        raw = results[self.output_names[0]]
        return postprocess_hef_single_class(raw, frame.shape[1], frame.shape[0], CONF_THRESH)


def load_detector(mode: str, detector_backend: str, onnx_path: Path, hef_path: Path):
    """
    detector_backend:
      - auto: old behavior
          picamera -> hef
          webcam/video -> onnx
      - onnx: force ONNX
      - hef:  force HEF
    """
    if detector_backend == "auto":
        resolved_backend = "hef" if mode == "picamera" else "onnx"
    else:
        resolved_backend = detector_backend

    if resolved_backend == "hef":
        return HefDetector(hef_path), "hef"
    elif resolved_backend == "onnx":
        return OnnxDetector(onnx_path), "onnx"
    else:
        raise ValueError(f"Unsupported detector backend: {resolved_backend}")


def load_classifier_session(path: Path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Classifier not found: {path}")
    print(f"Loading classifier: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


# ──────────────────────────────────────────────
# DETECTOR HELPERS
# ──────────────────────────────────────────────

def run_detector(detector, frame):
    detections = detector.infer(frame)
    return [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf, cls_id) in detections]


def get_largest_box(detections):
    if not detections:
        return None
    return max(detections, key=lambda d: max(0, d[2] - d[0]) * max(0, d[3] - d[1]))


def apply_padding(x1, y1, x2, y2, frame_w, frame_h, pad=PADDING):
    pad_x = int((x2 - x1) * pad)
    pad_y = int((y2 - y1) * pad)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_w, x2 + pad_x),
        min(frame_h, y2 + pad_y),
    )


# ──────────────────────────────────────────────
# CLASSIFIER HELPERS
# ──────────────────────────────────────────────

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_crop(crop):
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def run_classifier(session, frame_buffer, classes):
    if len(frame_buffer) < SEQUENCE_LENGTH:
        return None, 0.0

    seq = np.stack(list(frame_buffer), axis=0)[np.newaxis].astype(np.float32)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: seq})[0][0]
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))

    if pred_idx < 0 or pred_idx >= len(classes):
        return None, 0.0

    return classes[pred_idx], float(probs[pred_idx])


# ──────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────

STATE_COLORS = {
    "walk": (0, 255, 0),
    "stop": (0, 0, 255),
    "caution": (0, 165, 255),
    None: (128, 128, 128),
}


def draw_overlay(frame, box, state, confidence, fps, buffer_len, real_detection, detector_backend):
    display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    sx = DISPLAY_W / frame.shape[1]
    sy = DISPLAY_H / frame.shape[0]
    color = STATE_COLORS.get(state, (128, 128, 128))

    if box is not None:
        x1, y1, x2, y2, det_conf = box
        cv2.rectangle(
            display,
            (int(x1 * sx), int(y1 * sy)),
            (int(x2 * sx), int(y2 * sy)),
            color,
            2,
        )
        cv2.putText(
            display,
            f"det {det_conf:.2f}",
            (int(x1 * sx), max(20, int(y1 * sy) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    label = f"{state.upper()}  {confidence * 100:.0f}%" if state else "NO SIGNAL"
    cv2.rectangle(display, (0, 0), (DISPLAY_W, 78), (0, 0, 0), -1)
    cv2.putText(display, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(display, f"detector: {detector_backend}", (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.putText(display, f"{fps:.1f} fps",
                (DISPLAY_W - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(display, f"buffer {buffer_len}/{SEQUENCE_LENGTH}",
                (DISPLAY_W - 160, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if not real_detection and buffer_len > 0:
        cv2.putText(display, "repeat",
                    (DISPLAY_W - 160, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 200), 1)

    return display


# ──────────────────────────────────────────────
# VIDEO SOURCES
# ──────────────────────────────────────────────

def open_webcam():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    return cap


def open_video(path):
    cap = cv2.VideoCapture(str(Path(path).expanduser().resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def open_picamera():
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise RuntimeError("picamera2 not found. Install: sudo apt install python3-picamera2") from e

    class PiCameraSource:
        def __init__(self):
            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameRate": 30},
            )
            self.cam.configure(cfg)
            self.cam.start()
            time.sleep(0.5)

        def read(self):
            return True, cv2.cvtColor(self.cam.capture_array(), cv2.COLOR_RGB2BGR)

        def isOpened(self):
            return True

        def release(self):
            self.cam.stop()

    return PiCameraSource()


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def run(source, detector, detector_backend, classifier_session, classes,
        audio_mode="tones", save=False, save_path="output.mp4"):

    audio = AudioManager(mode=audio_mode)
    buffer = deque(maxlen=SEQUENCE_LENGTH)
    fps_acc = deque(maxlen=30)

    state = None
    confidence = 0.0
    box = None
    real_detection = False

    writer = None
    if save:
        save_path = Path(save_path)
        save_parent = save_path.parent if str(save_path.parent) != "" else Path(".")
        save_parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, 30.0, (DISPLAY_W, DISPLAY_H))
        print(f"Saving output to: {save_path}")

    print(f"Audio mode: {audio_mode}")
    print(f"Detector backend: {detector_backend}")
    print("Running — press Q to quit\n")

    while True:
        t0 = time.time()
        ret, frame = source.read()
        if not ret:
            print("\nEnd of video or camera disconnected.")
            break

        h, w = frame.shape[:2]

        detections = run_detector(detector, frame)
        best_box = get_largest_box(detections)

        if best_box is not None:
            x1, y1, x2, y2, det_conf = best_box
            x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2, w, h)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                buffer.append(preprocess_crop(crop))
                box = (x1, y1, x2, y2, det_conf)
                real_detection = True
            else:
                if len(buffer) > 0:
                    buffer.append(buffer[-1])
                box = None
                real_detection = False
        else:
            if len(buffer) > 0:
                buffer.append(buffer[-1])
            box = None
            real_detection = False

        if len(buffer) == SEQUENCE_LENGTH:
            state, confidence = run_classifier(classifier_session, buffer, classes)
            audio.update(state, real_detection)
            print(
                f"\r  {str(state):<10} {confidence*100:5.1f}%"
                f"  |  {'LIVE' if real_detection else 'rpt ':4s}"
                f"  |  box: {'yes' if box else 'no ':3s}"
                f"  |  {1/max(time.time()-t0,1e-6):5.1f} fps   ",
                end=""
            )
        else:
            print(f"\r  Buffering... {len(buffer)}/{SEQUENCE_LENGTH}"
                  f"  |  box: {'yes' if box else 'no ':3s}   ", end="")

        fps_acc.append(1.0 / max(time.time() - t0, 1e-6))
        display = draw_overlay(
            frame, box, state, confidence,
            float(np.mean(fps_acc)), len(buffer), real_detection, detector_backend
        )

        cv2.imshow("Pedestrian Signal", display)

        if writer is not None:
            writer.write(display)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    source.release()
    if writer is not None:
        writer.release()
        print(f"\nVideo saved to: {save_path}")
    cv2.destroyAllWindows()
    print("\nDone.")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pedestrian Signal Inference")

    parser.add_argument("--initialize", action="store_true",
                        help="Install Python dependencies from requirements.txt and exit")

    parser.add_argument("--mode", choices=["picamera", "webcam", "video"],
                        help="Inference source mode")
    parser.add_argument("--source", default=None,
                        help="Video file path (required for --mode video)")

    parser.add_argument("--detector-backend", default="auto",
                        choices=["auto", "onnx", "hef"],
                        help="Detector backend: auto = picamera->hef, webcam/video->onnx")

    parser.add_argument("--detector-onnx", default=str(DETECTOR_ONNX))
    parser.add_argument("--detector-hef", default=str(DETECTOR_HEF))

    parser.add_argument("--classifier", default=str(CLASSIFIER_ONNX))
    parser.add_argument("--classmap", default=str(CLASS_MAP_PATH))
    parser.add_argument("--requirements", default=str(REQUIREMENTS_PATH))

    parser.add_argument("--audio", default=DEFAULT_AUDIO_MODE,
                        choices=["tones", "tts"],
                        help="Audio output mode")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output video to file")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH,
                        help=f"Output video path (default: {DEFAULT_SAVE_PATH})")

    args = parser.parse_args()

    if args.initialize:
        initialize_environment(Path(args.requirements))
        return

    if not args.mode:
        parser.error("--mode is required unless using --initialize")

    if args.mode == "video" and not args.source:
        parser.error("--source is required for --mode video")

    classmap_path = Path(args.classmap).expanduser().resolve()
    if not classmap_path.exists():
        raise FileNotFoundError(f"class_map.json not found: {classmap_path}")

    with open(classmap_path, "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    detector, detector_backend = load_detector(
        args.mode,
        args.detector_backend,
        Path(args.detector_onnx).expanduser().resolve(),
        Path(args.detector_hef).expanduser().resolve(),
    )

    classifier_session = load_classifier_session(
        Path(args.classifier).expanduser().resolve()
    )

    if args.mode == "picamera":
        source = open_picamera()
    elif args.mode == "webcam":
        source = open_webcam()
    else:
        source = open_video(args.source)

    print(f"Classes:             {classes}")
    print(f"Torch device:        {TORCH_DEVICE}")
    print(f"Audio mode:          {args.audio}")
    print(f"Detector selection:  {args.detector_backend}")
    print(f"Detector mode:       {detector_backend}")
    print(f"Save video:          {args.save}" + (f" → {args.save_path}" if args.save else ""))

    run(
        source,
        detector,
        detector_backend,
        classifier_session,
        classes,
        audio_mode=args.audio,
        save=args.save,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
