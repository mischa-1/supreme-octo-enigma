import subprocess
import platform
import time

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
             from gtts import gTTS
            from playsound import playsound
            import os

            filename = "tts_output.mp3"

            tts = gTTS(text=text, lang="en")
            tts.save(filename)

            playsound(filename)

            # Optional: delete file after playing
            os.remove(filename)

    def cleanup(self):
        """No cleanup needed for native say command"""
        pass


def main():
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
    
    tts.cleanup()


if __name__ == "__main__":
    main()
