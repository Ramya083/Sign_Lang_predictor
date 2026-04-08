"""
collect_data.py — Collect real hand landmark data via webcam.
Run: python collect_data.py

Controls:
  Press letter key → collect that sign (100 samples)
  Press 'q'        → save and quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["del", "nothing", "space"]
SAMPLES_PER_LABEL = 100
DATA_PATH = "model/collected_data.pkl"


def collect():
    os.makedirs("model", exist_ok=True)

    dataset = {}
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded existing dataset.")
    else:
        dataset = {label: [] for label in LABELS}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    current_label = None
    collecting    = False
    session_count = 0

    print("\n=== ASL Data Collector ===")
    print("Press a letter key to start collecting that sign.")
    print("Press 'q' to quit and save.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            if collecting and current_label:
                lm   = results.multi_hand_landmarks[0]
                flat = []
                for pt in lm.landmark:
                    flat.extend([pt.x, pt.y, pt.z])
                dataset[current_label].append(np.array(flat))
                session_count += 1

                if len(dataset[current_label]) >= SAMPLES_PER_LABEL:
                    print(f"  Done: '{current_label}'")
                    collecting    = False
                    current_label = None

        status = f"Recording: {current_label} [{len(dataset.get(current_label,[]))}/{SAMPLES_PER_LABEL}]" if collecting else "Ready — press a key"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,100), 2)
        cv2.imshow("ASL Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key != 255:
            ch = chr(key).upper()
            if ch in LABELS:
                current_label = ch
                collecting    = True
                print(f"  Recording '{ch}'...")

    cap.release()
    cv2.destroyAllWindows()

    with open(DATA_PATH, "wb") as f:
        pickle.dump(dataset, f)
    print(f"\nSaved: {DATA_PATH}")
    print("Now retrain: python train_model.py")


if __name__ == "__main__":
    collect()