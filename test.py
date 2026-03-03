import torch
import cv2
import numpy as np
from picamera2 import Picamera2

print("Torch:", torch.__version__)
print("NumPy:", np.__version__)
print("CV2:", cv2.__version__)

picam2 = Picamera2()
print("Camera detected")

# this file jsut tests that everything is downloaded correctly in the
# virtual environment on the pi
