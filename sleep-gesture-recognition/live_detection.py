# Program to recognise sleep gestures in real time
# Then play a sound effect and send data to lights and other visual components of
# the installation
# Part of Hybrid Worlds Installation of Group 1
# author - Gabriela Ilieva


# IMPORTS
from __future__ import annotations

import os
import warnings
from pathlib import Path
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

# warning cleanup
warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*X does not have valid feature names.*",
)

# paths
MODEL = Path("model/sleep_gesture_svm.joblib")
AUDIO_PATH = Path("placeholder.wav")   # sound to play when sleep mode is initiated