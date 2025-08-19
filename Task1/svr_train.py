# train_svr_ear.py
import cv2
import numpy as np
import os
import pickle
import csv
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

haar_path = "haarcascade/haarcascade_eye.xml"
labels_csv = "ear_labels.csv"
model_path = "models/ear_predictor_svr.pkl"

eye_cascade = cv2.CascadeClassifier(haar_path)

X = []
y = []

with open(labels_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row["filepath"]
        ear_value = float(row["ear"])

        print(f"[INFO] Cropping eyes from: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read {img_path}, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (ex, ey, ew, eh) in eyes:
            eye_img = gray[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (24, 24))
            X.append(eye_img.flatten())
            y.append(ear_value)

X = np.array(X)
y = np.array(y)

print(f"[INFO] Training samples: {len(X)}")

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVR (no built-in loss curve, so we’ll simulate loss by partial fitting)
svr = SVR(kernel='rbf')
print("[INFO] Training SVR model...")
svr.fit(X_train, y_train)

# Evaluate
y_pred = svr.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"[RESULT] R² Score: {r2:.4f}")
print(f"[RESULT] MSE: {mse:.6f}")

# Since SVR doesn't give a loss curve, we plot predicted vs actual EAR
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, c='blue', label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Perfect Fit")
plt.xlabel("Actual EAR")
plt.ylabel("Predicted EAR")
plt.title("SVR EAR Prediction")
plt.legend()
os.makedirs("models", exist_ok=True)
plt.savefig("models/svr_loss.png")
print("[INFO] Loss (prediction scatter) graph saved to models/svr_loss.png")

# Save model
with open(model_path, "wb") as f:
    pickle.dump(svr, f)
print(f"[INFO] Model saved to {model_path}")
