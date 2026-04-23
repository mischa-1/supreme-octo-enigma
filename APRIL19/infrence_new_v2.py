#!/usr/bin/env python3
"""
Pedestrian Signal Inference — v4 (RGB + Motion channel)

OS flag controls runtime backend selection:

  --os mac   -> detector .pt + classifier .pt
  --os pi    -> detector .onnx + classifier .onnx

Examples:
  python3 infrence_new_v2.py --os mac --mode video --source /path/to/video.mov
  python3 infrence_new_v2.py --os mac --mode video --source /path/to/video.mov --headless
"""

import os
import cv2
import json
import time
import argparse
import threading
import struct
import tempfile
import subprocess
from collections import deque

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


# ──────────────────────────────────────────────
# DIRECT PATHS
# ──────────────────────────────────────────────

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# detector model files
DETECTOR_PT = ROOT / "detector.pt"
DETECTOR_ONNX = ROOT / "detector.onnx"
DETECTOR_HEF = ROOT / "detector.hef"

# classifier-related files under runs/
CLASSIFIER_DIR = ROOT / "runs" / "classifier"
CLASSMAP_PATH = CLASSIFIER_DIR / "class_map.json"
CLASSIFIER_PT = CLASSIFIER_DIR / "best.pt"
CLASSIFIER_ONNX = CLASSIFIER_DIR / "best.onnx"


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

CONF_THRESH = 0.30
PADDING = 0.20
SEQUENCE_LENGTH = 60
IMG_SIZE = 128
INPUT_CHANNELS = 4

DISPLAY_W = 960
DISPLAY_H = 540

AUDIO_MODE = "tones"
SAVE_PATH = "output.mp4"

TONE_REPEAT_INTERVAL = {
    "walk": 1.5,
    "caution": 1.0,
    "stop": 2.0,
}

TTS_REPEAT_INTERVAL = 1.5
SAMPLE_RATE = 44100

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FEATURE_DIM = 576
GRU_HIDDEN = 128
GRU_LAYERS = 1
NUM_CLASSES = 3


# ──────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────

def get_torch_device():
    if not HAS_TORCH:
        return "cpu"
    try:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


TORCH_DEVICE = get_torch_device()


def get_detector_device():
    if not HAS_TORCH:
        return "cpu"
    try:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


DETECTOR_DEVICE = get_detector_device()


# ──────────────────────────────────────────────
# PREPROCESSING — V4 RGB + MOTION
# ──────────────────────────────────────────────

def preprocess_crop_4ch(curr_bgr, prev_bgr=None):
    curr = cv2.resize(curr_bgr, (IMG_SIZE, IMG_SIZE))
    curr_rgb = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    curr_normalized = (curr_rgb - MEAN) / STD
    rgb_chw = curr_normalized.transpose(2, 0, 1)

    if prev_bgr is None:
        motion = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    else:
        prev = cv2.resize(prev_bgr, (IMG_SIZE, IMG_SIZE))
        prev_f = prev.astype(np.float32) / 255.0
        curr_f = curr.astype(np.float32) / 255.0

        diff = np.abs(curr_f - prev_f)
        motion = diff.mean(axis=2)
        motion = np.clip(motion * 3.0, 0.0, 1.0)

    motion_ch = motion[np.newaxis]
    return np.concatenate([rgb_chw, motion_ch], axis=0).astype(np.float32)


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


# ──────────────────────────────────────────────
# AUDIO
# ──────────────────────────────────────────────

def generate_tone(freq, duration, volume=0.6):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    fade = int(SAMPLE_RATE * 0.01)
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
        "walk": concat_tones([generate_tone(800, 0.12), generate_tone(800, 0.12)], gap_ms=60),
        "caution": concat_tones([generate_tone(520, 0.09), generate_tone(520, 0.09), generate_tone(520, 0.09)], gap_ms=40),
        "stop": generate_tone(280, 0.25),
    }


def play_tone_array(wave):
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
            subprocess.run(
                ["afplay", tmp],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                ["aplay", tmp],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    finally:
        os.unlink(tmp)


class AudioManager:
    def __init__(self, mode="tones"):
        self.mode = mode
        self.last_state = None
        self.last_played = 0.0

        self._lock = threading.Lock()
        self._playing = False
        self._tts_playing = False
        self._tts_process = None

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
        with self._lock:
            if self._tts_playing:
                return
            self._tts_playing = True

        def _run():
            try:
                if os.path.exists("/usr/bin/say"):
                    self._tts_process = subprocess.Popen(
                        ["say", text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._tts_process.wait()
                    return

                if self.tts_engine:
                    try:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                        return
                    except Exception:
                        pass
            finally:
                with self._lock:
                    self._tts_playing = False
                    self._tts_process = None

        threading.Thread(target=_run, daemon=True).start()

    def update(self, state, real_detection):
        if state is None:
            return

        now = time.time()
        state_changed = state != self.last_state

        if self.mode == "tones":
            interval = TONE_REPEAT_INTERVAL.get(state, 2.0)
            due_for_repeat = (now - self.last_played) >= interval

            if real_detection and (state_changed or due_for_repeat):
                with self._lock:
                    playing = self._playing
                if not playing:
                    self._play_async(self.tones[state])
                    self.last_played = now

        elif self.mode == "tts":
            due_for_repeat = (now - self.last_played) >= TTS_REPEAT_INTERVAL

            with self._lock:
                tts_busy = self._tts_playing

            if (state_changed or due_for_repeat) and not tts_busy:
                self._speak_async(state)
                self.last_played = now

        self.last_state = state


# ──────────────────────────────────────────────
# CLASSIFIER BACKENDS
# ──────────────────────────────────────────────

class OnnxClassifier:
    def __init__(self, onnx_path, classes):
        self.session = ort.InferenceSession(
            os.path.abspath(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.classes = classes

        print(f"ONNX classifier loaded: {onnx_path}")
        print(f"Input shape: {self.session.get_inputs()[0].shape}")

    def infer(self, buffer_4ch):
        if len(buffer_4ch) < SEQUENCE_LENGTH:
            return None, 0.0

        seq = np.stack(list(buffer_4ch), axis=0)[np.newaxis].astype(np.float32)
        logits = self.session.run(None, {self.input_name: seq})[0][0]
        probs = softmax(logits)
        idx = int(np.argmax(probs))

        if idx < 0 or idx >= len(self.classes):
            return None, 0.0

        return self.classes[idx], float(probs[idx])


class PtSignalClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, gru_hidden=GRU_HIDDEN, temperature=1.0):
        super().__init__()

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        first_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            4,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight
            new_conv.weight[:, 3] = first_conv.weight.mean(dim=1) * 0.01

        backbone.features[0][0] = new_conv

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.gru = nn.GRU(FEATURE_DIM, gru_hidden, num_layers=GRU_LAYERS, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )
        self.temperature = temperature

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        x = self.pool(self.features(x))
        x = x.view(b, s, FEATURE_DIM)
        x = self.dropout(x)
        _, h_n = self.gru(x)
        return self.classifier(h_n[-1]) / self.temperature


class PtClassifier:
    def __init__(self, pt_path, classes):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for --os mac")

        ckpt = torch.load(os.path.abspath(pt_path), map_location=TORCH_DEVICE)

        self.classes = classes
        self.device = torch.device(TORCH_DEVICE)

        self.model = PtSignalClassifier(
            num_classes=len(classes),
            gru_hidden=ckpt.get("gru_hidden", GRU_HIDDEN),
            temperature=ckpt.get("temperature", 1.0),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        print(f"PT classifier loaded: {pt_path}")
        print(f"Device: {self.device}")
        print(f"Temperature: {self.model.temperature:.2f}")

    @torch.no_grad()
    def infer(self, buffer_4ch):
        if len(buffer_4ch) < SEQUENCE_LENGTH:
            return None, 0.0

        seq = np.stack(list(buffer_4ch), axis=0)[np.newaxis].astype(np.float32)
        tensor = torch.from_numpy(seq).to(self.device)
        logits = self.model(tensor)[0]
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

        idx = int(np.argmax(probs))
        if idx < 0 or idx >= len(self.classes):
            return None, 0.0

        return self.classes[idx], float(probs[idx])


# ──────────────────────────────────────────────
# DETECTOR
# ──────────────────────────────────────────────

def load_detector(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Detector not found: {path}")
    return YOLO(path)


def run_detector(model, frame):
    results = model(frame, conf=CONF_THRESH, verbose=False, device=DETECTOR_DEVICE)[0]
    detections = []
    for b in results.boxes:
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        detections.append((x1, y1, x2, y2, conf))
    return detections


def get_largest_box(detections):
    if not detections:
        return None
    return max(detections, key=lambda d: max(0, d[2] - d[0]) * max(0, d[3] - d[1]))


def apply_padding(x1, y1, x2, y2, w, h):
    px = int((x2 - x1) * PADDING)
    py = int((y2 - y1) * PADDING)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


# ──────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────

STATE_COLORS = {
    "walk": (0, 255, 0),
    "stop": (0, 0, 255),
    "caution": (0, 165, 255),
    None: (128, 128, 128),
}


def draw_overlay(frame, box, state, confidence, fps, buf_len, real_det, os_mode):
    display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    sx, sy = DISPLAY_W / frame.shape[1], DISPLAY_H / frame.shape[0]
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
    cv2.rectangle(display, (0, 0), (DISPLAY_W, 58), (0, 0, 0), -1)
    cv2.putText(display, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(display, f"{fps:.1f} fps", (DISPLAY_W - 120, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(display, f"buf {buf_len}/{SEQUENCE_LENGTH}  [{os_mode}]",
                (DISPLAY_W - 210, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if not real_det and buf_len > 0:
        cv2.putText(display, "repeat", (DISPLAY_W - 80, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 200), 1)
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
    cap = cv2.VideoCapture(os.path.abspath(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {path}")
    return cap


def open_picamera():
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise RuntimeError("picamera2 not installed") from e

    class PiCam:
        def __init__(self):
            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameRate": 30}
            )
            self.cam.configure(cfg)
            self.cam.start()
            time.sleep(0.5)

        def read(self):
            return True, self.cam.capture_array()

        def isOpened(self):
            return True

        def release(self):
            self.cam.stop()

    return PiCam()


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def run(source, detector, classifier, classes, os_mode, save=False, save_path="output.mp4", headless=False):
    audio = AudioManager(mode=AUDIO_MODE)

    buffer_4ch = deque(maxlen=SEQUENCE_LENGTH)
    fps_acc = deque(maxlen=30)

    state = None
    confidence = 0.0
    box = None
    real_detection = False
    prev_crop_bgr = None

    writer = None
    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, 30.0, (DISPLAY_W, DISPLAY_H))
        print(f"Saving to: {save_path}")

    if not headless:
        cv2.namedWindow("Pedestrian Signal", cv2.WINDOW_NORMAL)

    print(f"OS mode:    {os_mode}")
    print(f"Audio:      {AUDIO_MODE}")
    print(f"Channels:   {INPUT_CHANNELS} (RGB + motion)")
    print(f"Headless:   {headless}")
    print("Running — press Q to quit, or Ctrl+C to stop cleanly\n")

    try:
        while True:
            t0 = time.time()
            ret, frame = source.read()
            if not ret:
                print("\nEnd of source.")
                break

            h, w = frame.shape[:2]

            detections = run_detector(detector, frame)
            best_box = get_largest_box(detections)

            if best_box is not None:
                x1, y1, x2, y2, det_conf = best_box
                x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2, w, h)
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    tensor_4ch = preprocess_crop_4ch(crop, prev_crop_bgr)
                    buffer_4ch.append(tensor_4ch)
                    prev_crop_bgr = crop.copy()
                    box = (x1, y1, x2, y2, det_conf)
                    real_detection = True
                else:
                    if buffer_4ch:
                        repeated = buffer_4ch[-1].copy()
                        repeated[3] = 0.0
                        buffer_4ch.append(repeated)
                    box = None
                    real_detection = False
            else:
                if buffer_4ch:
                    repeated = buffer_4ch[-1].copy()
                    repeated[3] = 0.0
                    buffer_4ch.append(repeated)
                box = None
                real_detection = False

            current_fps = 1.0 / max(time.time() - t0, 1e-6)
            fps_acc.append(current_fps)

            if len(buffer_4ch) == SEQUENCE_LENGTH:
                state, confidence = classifier.infer(buffer_4ch)
                audio.update(state, real_detection)
                print(
                    f"\r  {str(state):<10} {confidence * 100:5.1f}%"
                    f"  |  {'LIVE' if real_detection else 'rpt ':4s}"
                    f"  |  box: {'yes' if box else 'no ':3s}"
                    f"  |  {current_fps:5.1f} fps   ",
                    end=""
                )
            else:
                print(
                    f"\r  Buffering {len(buffer_4ch)}/{SEQUENCE_LENGTH}"
                    f"  |  box: {'yes' if box else 'no ':3s}"
                    f"  |  {current_fps:5.1f} fps   ",
                    end=""
                )

            display = None
            if (not headless) or writer is not None:
                display = draw_overlay(
                    frame, box, state, confidence,
                    float(np.mean(fps_acc)), len(buffer_4ch),
                    real_detection, os_mode
                )

            if not headless and display is not None:
                cv2.imshow("Pedestrian Signal", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break

            if writer is not None and display is not None:
                writer.write(display)

    except KeyboardInterrupt:
        print("\n\n[INFO] Ctrl+C detected — shutting down cleanly...")

    finally:
        try:
            source.release()
        except Exception:
            pass

        if writer is not None:
            writer.release()
            print(f"\nSaved: {save_path}")

        if not headless:
            cv2.destroyAllWindows()

        if fps_acc:
            print(f"\nAverage FPS: {float(np.mean(fps_acc)):.2f}")
            print(f"Max FPS:     {float(np.max(fps_acc)):.2f}")

        print("\nDone.")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    global AUDIO_MODE

    parser = argparse.ArgumentParser(description="Pedestrian Signal Inference v4")
    parser.add_argument("--os", required=True, choices=["mac", "pi"])
    parser.add_argument("--mode", required=True, choices=["picamera", "webcam", "video"])
    parser.add_argument("--source", default=None)
    parser.add_argument("--audio", default=AUDIO_MODE, choices=["tones", "tts"])
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-path", default=SAVE_PATH)
    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--detector", default=None)
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--classmap", default=CLASSMAP_PATH)

    args = parser.parse_args()
    AUDIO_MODE = args.audio

    if args.mode == "video" and not args.source:
        parser.error("--source required for --mode video")

    classmap_path = os.path.abspath(args.classmap)
    if not os.path.exists(classmap_path):
        raise FileNotFoundError(f"class_map.json not found: {classmap_path}")

    with open(classmap_path) as f:
        classes = json.load(f)["classes"]

    if args.os == "mac":
        detector_path = os.path.abspath(args.detector or DETECTOR_PT)
        classifier_path = os.path.abspath(args.classifier or CLASSIFIER_PT)
        classifier = PtClassifier(classifier_path, classes)
    else:
        detector_path = os.path.abspath(args.detector or DETECTOR_ONNX)
        classifier_path = os.path.abspath(args.classifier or CLASSIFIER_ONNX)
        classifier = OnnxClassifier(classifier_path, classes)

    print(f"Classes:      {classes}")
    print(f"OS mode:      {args.os}")
    print(f"Detector:     {detector_path}")
    print(f"Classifier:   {classifier_path}")
    print(f"Audio:        {AUDIO_MODE}")
    print(f"Headless:     {args.headless}")

    if not os.path.exists(detector_path):
        raise FileNotFoundError(f"Detector file not found: {detector_path}")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier file not found: {classifier_path}")

    detector = load_detector(detector_path)

    if args.mode == "picamera":
        source = open_picamera()
    elif args.mode == "webcam":
        source = open_webcam()
    else:
        source = open_video(args.source)

    run(
        source=source,
        detector=detector,
        classifier=classifier,
        classes=classes,
        os_mode=args.os,
        save=args.save,
        save_path=args.save_path,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
