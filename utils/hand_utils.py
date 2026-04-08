"""
utils/hand_utils.py
Handles MediaPipe hand detection and landmark extraction.
"""

import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_hands_detector(static_mode=True, max_hands=1, confidence=0.7):
    return mp_hands.Hands(
        static_image_mode=static_mode,
        max_num_hands=max_hands,
        min_detection_confidence=confidence,
        min_tracking_confidence=0.5,
    )


def extract_landmarks(image_rgb, hands_detector):
    """
    Returns (landmarks_flat, annotated_image, hand_detected)
    landmarks_flat: np.array of shape (63,) — 21 landmarks * [x,y,z]
    """
    results = hands_detector.process(image_rgb)
    annotated = image_rgb.copy()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            annotated,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        flat = []
        for lm in hand_landmarks.landmark:
            flat.extend([lm.x, lm.y, lm.z])

        return np.array(flat), annotated, True

    return None, annotated, False


def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist for translation invariance."""
    reshaped = landmarks.reshape(21, 3)
    wrist = reshaped[0].copy()
    reshaped -= wrist
    max_val = np.max(np.abs(reshaped)) + 1e-8
    reshaped /= max_val
    return reshaped.flatten()