# Program to recognise sleep gestures in real time
# Then play a sound effect and send data to lights and other visual components of
# the installation
# Part of Hybrid Worlds Installation of Group 1
# author - Gabriela Ilieva


# IMPORTS
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import Counter, deque

import serial

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
MODEL = BASE_DIR / "model" / "sleep_gesture_svm.joblib"

# serial settings
SERIAL_PORT = "/dev/cu.usbserial-10"  # placeholder: change later to actual device
SERIAL_BAUDRATE = 115200

serial_connection = None
last_received_packet = None

# packet settings
# the packet is expected to contain 5 integer values separated by commas
# this node modifies only the last value
PACKET_LENGTH = 5
SLEEP_INDEX = 4
SLEEP_VALUE = 1
DEFAULT_VALUE = -1

# settings
CAMERA_INDEX = 0
MAX_NUM_HANDS = 2

# confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

CONFIDENCE_THRESHOLD = 0.80
GESTURE_HOLD_SECONDS = 1.5
SLEEP_DURATION_SECONDS = 10.0
DEMO_KEY = ord("d") # demo day fallback

SMOOTHING_WINDOW = 5 # predictions to consider before committing to gesture
TRIGGER_COOLDOWN = 2.0 # prevent rapid flickering



CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# labels for gestures/poses
LABELS = [
    "neutral",
    "sleep_left",
    "sleep_right",
    "hand_near_face_no_sleep",
    "head_lean_only",
]

FEATURE_COLUMNS = []

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

# feature extraction like in the data collection script
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

# prediction function that returns label + confidence
def predict_pose_label(features_df: pd.DataFrame, model) -> tuple[str | None, float]:
    probs = model.predict_proba(features_df)[0]
    best_index = np.argmax(probs)
    confidence = probs[best_index]
    pred = model.classes_[best_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence

    if pred in LABELS:
        return pred, confidence

    return None, confidence

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

# connect to the hardware over serial
def connect_serial():
    global serial_connection

    try:
        serial_connection = serial.Serial(
            SERIAL_PORT,
            SERIAL_BAUDRATE,
            timeout=1
        )

        # arduino resets after connection
        time.sleep(2)

        print(f"Connected to serial device on {SERIAL_PORT}")

    except Exception as e:
        serial_connection = None
        print(f"Could not connect to serial device: {e}")

# receive one packet from serial
def receive_serial_packet():
    global serial_connection

    if serial_connection is None:
        return None

    try:
        if serial_connection.in_waiting > 0:
            packet = serial_connection.readline().decode("utf-8").strip()

            if packet:
                print(f"Received packet: {packet}")
                return packet

    except Exception as e:
        print(f"Serial receive error: {e}")

    return None

def format_packet_value(value):
    return f"{value:.2f}"

def update_sleep_state_in_packet(packet, is_asleep):
    try:
        values = [float(x.strip()) for x in packet.split(",")]

        if len(values) != PACKET_LENGTH:
            print(f"Invalid packet length: {packet}")
            return packet

        old_value = values[SLEEP_INDEX]

        if is_asleep:
            values[SLEEP_INDEX] = float(SLEEP_VALUE)
        else:
            values[SLEEP_INDEX] = float(DEFAULT_VALUE)

        print(
            f"Sleep value changed from {old_value} "
            f"to {values[SLEEP_INDEX]}"
        )

        # keep packet format nice
        return ",".join(format_packet_value(v) for v in values)

    except Exception as e:
        print(f"Could not update packet: {e}")
        return packet
# send text commands through serial
def send_serial_command(command):
    global serial_connection

    # handle no connection
    if serial_connection is None:
        print(f"[SERIAL NOT CONNECTED] Would send: {command}")
        return

    # send message
    try:
        message = f"{command}\n"
        serial_connection.write(message.encode())

        print(f"Sent: {command}")

    # handle serial errors
    except Exception as e:
        print(f"Serial error: {e}")

def send_sleep_state_immediately(is_asleep):
    global last_received_packet

    if last_received_packet is None:
        print("No previous packet available yet, cannot send immediate sleep update.")
        return

    outgoing_packet = update_sleep_state_in_packet(
        last_received_packet,
        is_asleep
    )

    print(f"Immediate sleep update: {outgoing_packet}")
    send_serial_command(outgoing_packet)

def handle_incoming_packet(is_asleep):
    global last_received_packet

    incoming_packet = receive_serial_packet()

    if incoming_packet is None:
        return

    last_received_packet = incoming_packet

    outgoing_packet = update_sleep_state_in_packet(
        incoming_packet,
        is_asleep
    )

    print(f"Forwarding packet: {outgoing_packet}")

    send_serial_command(outgoing_packet)

# main
def main() -> None:
    # check paths
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    # load model
    model = joblib.load(MODEL)

    # connect to serial
    connect_serial()

    # start webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # set resolution BEFORE reading frames
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # don't use buffersize on Mac for now
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # warm up camera
    print("Warming up camera...")
    frame = None

    for i in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        cap.release()
        raise RuntimeError("Camera opened, but failed to read frames after warmup.")

    # stable prediction
    stable_pred = StablePrediction(window_size=SMOOTHING_WINDOW)

    # sleep/awake state variables
    current_state = "awake"

    sleep_candidate_start = None
    sleep_started_time = None

    last_triggered = None
    last_trigger_time = 0.0

    # helper function for changing state
    def change_state(new_state):
        nonlocal current_state, sleep_started_time

        if current_state == new_state:
            return

        print(f"STATE CHANGE: {current_state} -> {new_state}")
        current_state = new_state

        if new_state == "asleep":
            sleep_started_time = time.time()
            print("ACTION: Sleep transition initiated")
            send_sleep_state_immediately(True)

        elif new_state == "awake":
            print("ACTION: Sleep transition ended")
            send_sleep_state_immediately(False)


    # prints
    print("Starting sleep gesture detection...")
    print("Press 'q' to quit | 'd' for demo")

    # setup skeleton overlay
    hands = mp_hands.Hands(
        static_image_mode=False,  # use tracking (video mode)
        max_num_hands=MAX_NUM_HANDS,  # collect both hands
        # confidence thresholds
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    # create head detector object
    pose = mp_pose.Pose(
        static_image_mode=False,
        # confidence thresholds
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    failed_reads = 0
    MAX_FAILED_READS = 10
    # reading from camera and making predictions
    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            failed_reads += 1
            print(f"Failed to read from camera ({failed_reads}/{MAX_FAILED_READS})")
            time.sleep(0.1)

            if failed_reads >= MAX_FAILED_READS:
                print("Too many camera read failures. Exiting.")
                break

            continue
        failed_reads = 0

        # receive serial packet and forward it with updated sleep state
        handle_incoming_packet(current_state == "asleep")

        # keep the flip (data collection also uses this)
        frame = cv2.flip(frame, 1)

        # convert for mediapipe from bgr to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        # draw all detected hands
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # draw pose landmarks
        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # extract full two-hand + pose features
        features = extract_features(hand_results, pose_results)

        current_prediction = None
        confidence = 0.0

        if features is not None:
            features_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            raw_pred, confidence = predict_pose_label(features_df, model)
            current_prediction = stable_pred.update(raw_pred)
        else:
            stable_pred.clear()

        # display prediction
        if current_prediction is None:
            display_text = "Prediction: no pose detected"
        else:
            display_text = f"Prediction: {current_prediction}"

        cv2.putText(
            frame,
            display_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # trigger sound effect when sleep gesture is detected
        now = time.time()

        #trigerring sleep gesture if held for a specific time
        if current_state == "awake":
            if current_prediction in ["sleep_left", "sleep_right"]:
                if sleep_candidate_start is None:
                    sleep_candidate_start = now

                held_time = now - sleep_candidate_start

                if held_time >= GESTURE_HOLD_SECONDS and now - last_trigger_time > TRIGGER_COOLDOWN:
                    print(f"ACTION: Sleep gesture held: {current_prediction}")
                    change_state("asleep")
                    last_triggered = current_prediction
                    last_trigger_time = now
                    sleep_candidate_start = None

            else:
                sleep_candidate_start = None
        if current_state == "asleep":
            if sleep_started_time is not None and now - sleep_started_time >= SLEEP_DURATION_SECONDS:
                change_state("awake")
                sleep_started_time = None
                last_triggered = None
                sleep_candidate_start = None
        if current_prediction not in ["sleep_left", "sleep_right"]:
            last_triggered = None

        cv2.putText(
            frame,
            "Press 'q' to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Sleep Gesture Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        elif key == DEMO_KEY:
            print("DEMO MODE: forced sleep transition")
            change_state("asleep")

    cap.release()
    cv2.destroyAllWindows()

    if serial_connection is not None:
        serial_connection.close()
        print("Serial connection closed.")


if __name__ == "__main__":
    main()