# Script to collect data from mediapipe gestures
# Numbers 1-5 are used to start automatic burst recording for hands-free data collection
# of each pose/gesture

# imports
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time


# set output file name
OUTPUT_FILE = "sleep_gesture_data.csv"

# burst settings for hands-free data collection
COUNTDOWN_SECONDS = 5
RECORD_SECONDS = 5
SAMPLES_PER_SECOND = 10
SAMPLE_INTERVAL = 1 / SAMPLES_PER_SECOND


# MediaPipe setup
mp_hands = mp.solutions.hands #hand detection model
mp_draw = mp.solutions.drawing_utils # landmarks overlay on screen
mp_pose = mp.solutions.pose # head position

# create hand detector object
hands = mp_hands.Hands(
    static_image_mode=False, # use tracking (video mode)
    max_num_hands=2,  # collect both hands
    # confidence thresholds
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# create head detector object
pose = mp_pose.Pose(
    static_image_mode = False,
    # confidence thresholds
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# open default webcam (at index 0)
cap = cv2.VideoCapture(0)

# handle camera not able to start
if not cap.isOpened():
    print("Could not open camera.")
    exit()

# set up csv
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

# build header row (63 feature columns + 1 label column)
feature_columns = []

# collect left and right hand position
for side in ["left", "right"]:
    for i in range(21):
        feature_columns.extend([
            f"{side}_x{i}", f"{side}_y{i}", f"{side}_z{i}"
        ])

# collect head position through each pose point
pose_points = ["nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder"]
for point in pose_points:
    feature_columns.extend([
        f"{point}_x",
        f"{point}_y",
        f"{point}_z"
    ])

feature_columns.append("label")

# create CSV only if it doesn't already exist so the script can be used multiple times to keep
# adding data without duplicating the header
if not os.path.exists(OUTPUT_FILE):
    df_init = pd.DataFrame(columns=feature_columns)
    df_init.to_csv(OUTPUT_FILE, index=False)

current_label = None
record_count = 0

# print controls for easier collection
print("\nControls:")
print("  1 = neutral")
print("  2 = sleep_left")
print("  3 = sleep_right")
print("  4 = hand_near_face_no_sleep")
print("  5 = head_lean_only")
print("  Press a number to start 5s countdown + automatic recording")
print("  q = quit\n")

# convert keys to ascii for organised way to set strings to keys
label_map = {
    ord('1'): "neutral",
    ord('2'): "sleep_left",
    ord('3'): "sleep_right",
    ord('4'): "hand_near_face_no_sleep",
    ord('5'): "head_lean_only",
}

# helper function to draw the current MediaPipe detections during countdown/recording
def draw_detections(frame, hand_results, pose_results):
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

# helper function to collect a burst of samples after a countdown
def auto_record(label):
    global record_count

    print(f"\nGet ready for: {label}")
    print(f"Recording starts in {COUNTDOWN_SECONDS} seconds...")

    # countdown before recording so you can get into the gesture position
    countdown_start = time.time()

    while time.time() - countdown_start < COUNTDOWN_SECONDS:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read from camera.")
            return

        # flip camera horizontally to make easier to use
        frame = cv2.flip(frame, 1)
        # since opencv reads in bgr, we need to convert to rgb for mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        draw_detections(frame, hand_results, pose_results)

        remaining = COUNTDOWN_SECONDS - int(time.time() - countdown_start)

        cv2.putText(frame, f"Get ready: {label}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, f"Recording in: {remaining}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, f"Total saved: {record_count}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return

    print(f"Recording {label}...")

    # record multiple samples automatically
    samples_saved = 0
    failed_samples = 0
    last_sample_time = 0
    record_start = time.time()

    while time.time() - record_start < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read from camera.")
            return

        # flip camera horizontally to make easier to use
        frame = cv2.flip(frame, 1)
        # since opencv reads in bgr, we need to convert to rgb for mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        draw_detections(frame, hand_results, pose_results)

        now = time.time()

        if now - last_sample_time >= SAMPLE_INTERVAL:
            # extract full two-hand + pose features
            features = extract_features(hand_results, pose_results)

            if features is not None:
                row = list(features) + [label]
                df_row = pd.DataFrame([row], columns=feature_columns)
                df_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False) # mode is append, header shouldn't be duplicated
                record_count += 1
                samples_saved += 1
            else:
                failed_samples += 1

            last_sample_time = now

        time_left = RECORD_SECONDS - int(time.time() - record_start)

        cv2.putText(frame, f"Recording: {label}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Time left: {time_left}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Saved this round: {samples_saved}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Failed this round: {failed_samples}", (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return

    print(f"Done recording {label}. Saved {samples_saved} samples. Failed {failed_samples} samples.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read from camera.")
        break

    # flip camera horizontally to make easier to use
    frame = cv2.flip(frame, 1)
    # since opencv reads in bgr, we need to convert to rgb for mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    # text display
    display_text = f"Label: {current_label if current_label else 'None'} | Saved: {record_count}"
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    features = None

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

    cv2.imshow("Gesture Data Collection", frame)

    # mask the key returned to limit to the last 8 bits as sometimes it returns garbage
    key = cv2.waitKey(1) & 0xFF

    # change label
    if key in label_map:
        current_label = label_map[key]
        print(f"Current label set to: {current_label}")
        auto_record(current_label)

    # quit
    elif key == ord('q'):
        break

# close program by closing the webcam and closing the window
cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Data saved to {OUTPUT_FILE}") # final print