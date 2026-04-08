"""
train_model.py
Run this ONCE to train and save the ASL classifier.
Usage: python train_model.py
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["del", "nothing", "space"]

def generate_synthetic_landmarks(label, n_samples=150):
    np.random.seed(hash(label) % (2**32))
    samples = []
    base = np.random.rand(63) * 0.5 + 0.25
    for _ in range(n_samples):
        noise = np.random.normal(0, 0.015, 63)
        sample = np.clip(base + noise, 0, 1)
        samples.append(sample)
    return np.array(samples)

def build_dataset():
    X, y = [], []
    for label in LABELS:
        data = generate_synthetic_landmarks(label)
        X.extend(data)
        y.extend([label] * len(data))
    return np.array(X), np.array(y)

def train():
    print("Building dataset...")
    X, y = build_dataset()
    print(f"  Total samples: {len(X)}, Labels: {len(LABELS)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  Validation accuracy: {acc*100:.1f}%")

    os.makedirs("model", exist_ok=True)
    with open("model/asl_classifier.pkl", "wb") as f:
        pickle.dump({"model": clf, "labels": LABELS}, f)

    print("  Saved to model/asl_classifier.pkl")
    print("\nDone! Now run: streamlit run app.py")

if __name__ == "__main__":
    train()