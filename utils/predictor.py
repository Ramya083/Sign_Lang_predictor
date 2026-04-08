import pickle
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ASLPredictor:
    def __init__(self, model_path="model/asl_classifier.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Please run: python train_model.py"
            )
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.labels = data["labels"]

    def predict(self, landmarks: np.ndarray):
        x = landmarks.reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        idx = np.argmax(probs)
        label = self.labels[idx]
        confidence = probs[idx]

        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(self.labels[i], round(float(probs[i]) * 100, 1)) for i in top3_idx]

        return label, float(confidence), top3