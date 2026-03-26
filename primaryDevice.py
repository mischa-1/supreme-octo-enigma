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
def parse_arguments():
    parser = argparse.ArgumentParser(description= "Date for this program")
    parser.add_argument(
            "--TTS",
            action="store_true",
            help="Test that TTS is working on the Pi with basic example"
        )
    return parser.parse_args()

##### Run TOFL *** check with Rachel

#### AI HAT Set up
#def hailo_set_up():
    # to set up

##### Run YOLO

    

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
