##### Import Libaries
import time
import numpy as np
import argparse
from pathlib import Path

from picamera2 import Picamera2
import cv2

import hailo_platform as hpf
import subprocess
import platform

from gtts import gTTS
#from playsound import playsound

##### Create arpsgarse (debug mode)
# debug mode
# airpod address

# ---- CURRENT MODEL ----
MODEL_NAME = "yolo26n-1.hef"

def parse_arguments():
    parser = argparse.ArgumentParser(description= "Date for this program")
    parser.add_argument(
            "--TTS",
            action="store_true",
            help="Test that TTS is working on the Pi with basic example"
        )
    parser.add_argument(
            "--silent",
            action="store_true",
            help="Run code in silent mode for debugging purpouses"
        )
    parser.add_argument(
        "--hailo",
        action="store_true",
        help="Enable Hailo debug output"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print Hazard detections"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Number of classes in the model"
    )
    return parser.parse_args()

##### Run TOFL *** check with Rachel

##### Camera Set Up
def setup_camera(in_w, in_h):
    """Start camera at model input size."""
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (in_w, in_h), "format": "RGB888"}
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)

    return picam2

#### AI HAT Set up
def setup_hailo():
    """Load HEF and prepare Hailo device + streams."""

    model_path = Path("models") / MODEL_NAME
    hef = hpf.HEF(str(model_path))

    vdevice = hpf.VDevice()

    cfg = hpf.ConfigureParams.create_from_hef(
        hef, interface=hpf.HailoStreamInterface.PCIe
    )

    network_group = vdevice.configure(hef, cfg)[0]
    ng_params = network_group.create_params()

    in_info = hef.get_input_vstream_infos()[0]
    out_infos = hef.get_output_vstream_infos()

    in_params = hpf.InputVStreamParams.make_from_network_group(
        network_group, quantized=True, format_type=hpf.FormatType.UINT8
    )
    out_params = hpf.OutputVStreamParams.make_from_network_group(
        network_group, quantized=True, format_type=hpf.FormatType.UINT8
    )

    return vdevice, network_group, ng_params, in_info, out_infos, in_params, out_params

##### Run YOLO
def run_YOLO(
    picam2,
    network_group,
    ng_params,
    in_params,
    out_params,
    in_info,
    num_classes=3,
    score_thresh=225,
    min_hot_cells=1,
    hailo_debug=False
):
    """Return detection + class index using class tensors."""

    in_h, in_w, _ = in_info.shape

    with network_group.activate(ng_params):
        with hpf.InferVStreams(network_group, in_params, out_params) as pipe:
            frame = picam2.capture_array()
            resized = cv2.resize(frame, (in_w, in_h))

            input_data = {
                in_info.name: np.expand_dims(resized, axis=0)
            }

            outputs = pipe.infer(input_data)

    # ===== Find class tensors dynamically =====
    class_maps = []
    for name, arr in outputs.items():
        arr = np.asarray(arr)
        arr = np.squeeze(arr)

        if arr.ndim == 3 and arr.shape[-1] == num_classes:
            class_maps.append((name, arr))

    if not class_maps:
        if hailo_debug:
            print("No class tensors found.")
        return False, None

    # ===== Aggregate across scales =====
    class_scores = np.zeros(num_classes, dtype=np.float32)
    hot_cells_per_class = np.zeros(num_classes, dtype=int)

    for name, arr in class_maps:
        scale_max = np.max(arr, axis=(0, 1))
        class_scores = np.maximum(class_scores, scale_max)

        for c in range(num_classes):
            hot_cells_per_class[c] += int(np.count_nonzero(arr[:, :, c] >= score_thresh))

        if hailo_debug:
            print(f"\n{name} shape={arr.shape}")
            print("  per-class max:", scale_max)

    class_id = int(np.argmax(class_scores))
    confidence = float(class_scores[class_id])
    hot_cells = int(hot_cells_per_class[class_id])

    if hailo_debug:
        print("\nFinal class scores:", class_scores)
        print("Hot cells per class:", hot_cells_per_class)
        print("Chosen class:", class_id)
        print("Confidence:", confidence)
        print("Hot cells:", hot_cells)

    if confidence >= score_thresh and hot_cells >= min_hot_cells:
        return True, class_id
    else:
        return False, None
##### Airpod warning
# This code came from Tasha earlier in the semester

class TextToSpeech:
    def __init__(self, filename="tts_output.mp3"):
        self.player_proc = None
        self.filename = filename

    def speak(self, text: str):
        if not text:
            return

        # Stop previous playback if it is still running
        if self.player_proc and self.player_proc.poll() is None:
            self.player_proc.terminate()
            self.player_proc.wait()

        # Generate fresh mp3 from text
        tts = gTTS(text=text)
        tts.save(self.filename)

        # Play mp3 without blocking the rest of the program
        self.player_proc = subprocess.Popen(["mpg123", "-q", self.filename])

    def cleanup(self):
        if self.player_proc and self.player_proc.poll() is None:
            self.player_proc.terminate()
            self.player_proc.wait()


##### Talk to Pi ** check with Rachel


##### main
def main():
    args = parse_arguments()
    tts = TextToSpeech()

    # Setup AI HAT
    vdevice, network_group, ng_params, in_info, out_infos, in_params, out_params = setup_hailo()

    # Set Up Camera
    in_h, in_w, _ = in_info.shape
    picam2 = setup_camera(in_w, in_h)

    #----------------TEST ONE FRAME----------------
    if args.hailo:
        # Run YOLO
        detected, class_id = run_YOLO(
            picam2,
            network_group,
            ng_params,
            in_params,
            out_params,
            in_info,
            hailo_debug=args.hailo
        )
        
        if detected:
            print(f"Detected class {class_id}")
            if not args.silent:
                tts.speak(f"Hazard detected class {class_id}")
                time.sleep(2.5)
        else:
            print("No detection")

    # ===== YOLO Detection Loop =====
    try:
        while True:
            # --- Run YOLO ---
            detected, class_id = run_YOLO(
                picam2,
                network_group,
                ng_params,
                in_params,
                out_params,
                in_info,
                num_classes=args.num_classes,
                hailo_debug=args.hailo
            )
    
            # --- Handle Result ---
            if not detected:
                if args.print:
                    print("No detection")
            else:
                if args.print:
                    print(f"Detected class {class_id}")
    
                if not args.silent:
                    tts.speak(f"Hazard detected class {class_id}")
                    time.sleep(2.5)
    
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        # Optional cleanup
        try:
            tts.cleanup()
        except:
            pass

    ## Debugging
    if args.TTS:
        hazards = [
            "Person approaching from the right",
            "Obstacle directly ahead",
            "Clear path",
            "No hazards detected"
        ]

        for hazard in hazards:
            print(f"TTS Output: {hazard}")
            tts.speak(hazard)
            time.sleep(2.5)  # give each phrase time to play

    tts.cleanup()


if __name__ == "__main__":
    main()
