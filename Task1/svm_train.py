import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# Haar cascades
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")

# Dataset paths
dataset_path = "data"
categories = ["drowsy", "non_drowsy"]  # 1=open, 0=closed

features = []
labels = []

print("[INFO] Starting training data processing...")

for label, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    print(f"\n[INFO] Processing category: '{category}' (label={label})")

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(25, 10),   # width > height
                maxSize=(80, 40)    # width > height
            )

            # Keep only eyes in the upper half of face
            filtered_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey + eh/2 < h/2]

            for i, (ex, ey, ew, eh) in enumerate(filtered_eyes, start=1):
                ear = eh / ew
                print(f"[LOG] {img_name} | Eye {i} EAR={ear:.3f} | Label={category.upper()}")
                features.append([ear])
                labels.append(label)

print(f"\n[INFO] Total samples collected: {len(features)}")
print(f"[INFO] Samples per class: open={labels.count(1)}, closed={labels.count(0)}")

# Train SVM
print("\n[INFO] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

print(f"[INFO] Training set size: {len(X_train)}, Test set size: {len(X_test)}")
print("[INFO] Training SVM model...")
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

# Evaluate
print("[INFO] Evaluating model...")
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=categories))

# Save model
joblib.dump(svm_model, "models/haar_ear_svm.pkl")
print("[INFO] Model saved as haar_ear_svm.pkl")
