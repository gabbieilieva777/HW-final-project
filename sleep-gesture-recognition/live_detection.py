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
from collections import Counter, deque
import threading
import sounddevice as sd
import soundfile as sf

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
BASE_DIR = Path(__file__).resolve().parent
MODEL = BASE_DIR / "sleep_gesture_svm.joblib"
AUDIO_PATH = BASE_DIR / "sleepy_sound.wav"   # sound to play when sleep mode is initiated

# settings
CAMERA_INDEX = 0
MAX_NUM_HANDS = 2

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

SMOOTHING_WINDOW = 5 # predictions to consider before committing to gesture
TRIGGER_COOLDOWN = 0.35 # prevent rapid flickering

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# labels for gestures/poses
LABELS = [
    "neutral",
    "sleep_left",
    "sleep_right",
    "hand_near_face_no_sleep",
    "head_lean_only",
]
FEATURE_COLUMNS = []
for i in range(21):
    FEATURE_COLUMNS.extend([f"x{i}", f"y{i}", f"z{i}"])

# collect left and right hand position
for side in ["left", "right"]:
    for i in range(21):
        FEATURE_COLUMNS.extend([
            f"{side}_x{i}", f"{side}_y{i}", f"{side}_z{i}"
        ])

# collect head position through each pose point
pose_points = ["nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder"]
for point in pose_points:
    FEATURE_COLUMNS.extend([
        f"{point}_x",
        f"{point}_y",
        f"{point}_z"
    ])

# stable prediction helper class using a double-ended queue
class StablePrediction:
    # constructor
    def __init__(self, window_size: int = 5):
        # we use a deque to make the shifting of elements faster than a list and to have the max length
        self.history = deque(maxlen=window_size) # it automatically drops the last element when a new one enters

    # append new prediction each frame
    def update(self, label: str | None) -> str | None:
        if label is not None:
            self.history.append(label)

        if not self.history:
            return None
        # count how many times each label appears
        counts = Counter(self.history)
        return counts.most_common(1)[0][0] # return the prediction that appears most

    def clear(self) -> None:
        self.history.clear()

# feature extraction like in the data collection scrip
def extract_features(hand_results, pose_results):
    features = []

    # empty placeholders for both hands
    hand_data = {
        "Left": np.zeros(63, dtype=np.float32),
        "Right": np.zeros(63, dtype=np.float32)
    }

    # extract both hands
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness
        ):
            hand_side = handedness.classification[0].label  # "Left" or "Right"

            # collect hand coordinates
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            hand_data[hand_side] = np.array(coords, dtype=np.float32)

    else:
        return None  # no hands detected

    # allow recording if at least one hand is detected
    # this is important because the sleep gesture often hides/overlaps one hand
    if np.all(hand_data["Left"] == 0) and np.all(hand_data["Right"] == 0):
        return None

    # extract useful pose/head landmarks
    if not pose_results.pose_landmarks:
        return None

    pose_landmarks = pose_results.pose_landmarks.landmark

    # define pose points that we use to detect pose
    pose_indices = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
    ]

    # collect coordinates of those pose points
    pose_coords = []
    for idx in pose_indices:
        lm = pose_landmarks[idx]
        pose_coords.extend([lm.x, lm.y, lm.z])

    pose_coords = np.array(pose_coords, dtype=np.float32)

    # normalize everything relative to shoulder center
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    center_x = (left_shoulder.x + right_shoulder.x) / 2
    center_y = (left_shoulder.y + right_shoulder.y) / 2
    center_z = (left_shoulder.z + right_shoulder.z) / 2


    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if shoulder_width == 0:
        shoulder_width = 1

    # normalize hands
    for side in ["Left", "Right"]:
        coords = hand_data[side].copy()
        coords[0::3] = (coords[0::3] - center_x) / shoulder_width
        coords[1::3] = (coords[1::3] - center_y) / shoulder_width
        coords[2::3] = (coords[2::3] - center_z) / shoulder_width
        features.extend(coords)

    # normalize pose/head points
    pose_coords[0::3] = (pose_coords[0::3] - center_x) / shoulder_width
    pose_coords[1::3] = (pose_coords[1::3] - center_y) / shoulder_width
    pose_coords[2::3] = (pose_coords[2::3] - center_z) / shoulder_width

    features.extend(pose_coords)

    return np.array(features, dtype=np.float32)

# prediction
def predict_pose_label(features_df: pd.DataFrame, model) -> str | None:
    pred = model.predict(features_df)[0]
    return pred


# draw a label near head
def draw_label(frame, text: str, x: int, y: int, color=(0, 255, 0)) -> None:
    box_w = 280
    # clamping to prevent label from being drawn outside boundaries
    x = max(0, min(x, frame.shape[1] - box_w))
    y = max(35, min(y, frame.shape[0] - 10))
    cv2.rectangle(frame, (x, y - 28), (x + box_w, y + 8), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
def play_sound_effect():
    def _play():
        try:
            data, samplerate = sf.read(str(AUDIO_PATH), dtype="float32")
            sd.play(data, samplerate)
        except Exception as e:
            print(f"Could not play sound effect: {e}")

    threading.Thread(target=_play(), daemon=True).start()


def initiate_sleep_transition():
    if predict_pose_label() in ["sleep_left", "sleep_right"]:
        play_sound_effect()

# main
def main() -> None:
    global audio_engine
    # check paths
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}\n"
            f"Put a WAV file named 'sleepy_sound.wav' next to this script, or change AUDIO_PATH."
        )
    # load model
    model = joblib.load(MODEL)

    # draw skeleton
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # start webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # ssk camera for a normal resolution instead of stretching frames
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # help reduce camera buffering lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # stable prediction
    stable_pred = StablePrediction(window_size=SMOOTHING_WINDOW)

    # prints
    print("Starting sleep gesture detection...")
    print("Press 'q' to quit.")

    # setup skeleton overlay

    hands = mp_hands.Hands(
        static_image_mode=False,  # use tracking (video mode)
        max_num_hands=2,  # collect both hands
        # confidence thresholds
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # create head detector object
    pose = mp_pose.Pose(
        static_image_mode=False,
        # confidence thresholds
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # reading from camera and making predictions
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read from camera.")
            break

        # keep the flip (data collection also uses this)
        frame = cv2.flip(frame, 1)

        # convert for mediapipe from bgr to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        # extract both hands
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness
            ):
                hand_side = handedness.classification[0].label  # "Left" or "Right"


        else:
            return None  # no hands detected

        # extract useful pose/head landmarks
        if not pose_results.pose_landmarks:
            return None

        pose_landmarks = pose_results.pose_landmarks.landmark
