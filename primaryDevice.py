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
    def __init__(self, rate=160, volume=1.0):
        self.is_mac = platform.system() == "Darwin"
        self.rate = rate
        self.volume = volume

    def speak(self, text: str):
        if not text:
            return

        if self.is_mac:
            # Use macOS native 'say' command
            subprocess.run(['say', text], check=True)
        else:
            # Fallback to pyttsx3 for other platforms
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()

    def cleanup(self):
        """No cleanup needed for native say command"""
        pass


##### Talk to Pi ** check with Rachel


##### main
def main():

    # Setting things up
    args = parse_arguments()
    tts = TextToSpeech()

    if args.TTS:

        # This is gonna be debug text to speech thing
        tts = TextToSpeech()
    
        hazards = [
            "Person approaching from the right",
            "Obstacle directly ahead",
            "Clear path",
            "No hazards detected"
        ]
    
        for hazard in hazards:
            print(f"TTS Output: {hazard}")
            tts.speak(hazard)
            time.sleep(0.3)

    # Cleaning things up SAFELY
    tts.cleanup()


if __name__ == "__main__":
    main()
