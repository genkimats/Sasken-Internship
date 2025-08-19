import os
import pickle
import cv2
import dlib
import numpy as np
from imutils import face_utils
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_PATH = "models/svc_ear.pkl"
TEST_DIR = "custom_data"
LABELS = ["non_drowsy", "drowsy"]
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

X_test = []
y_test = []

for label_index, label_name in enumerate(LABELS):
    folder_path = os.path.join(TEST_DIR, label_name)
    if not os.path.isdir(folder_path):
        print(f"⚠ Skipping missing folder: {folder_path}")
        continue

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)
        if image is None:
            print(f"⚠ Could not read {file_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) == 0:
            print(f"⚠ No face detected in {file_path}")
            continue

        shape = predictor(gray, rects[0])
        shape_np = face_utils.shape_to_np(shape)

        leftEye = shape_np[42:48]
        rightEye = shape_np[36:42]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        X_test.append([ear])
        y_test.append(label_index)

print(f"Loaded {len(X_test)} samples from '{TEST_DIR}'")

X_test = np.array(X_test)
y_test = np.array(y_test)

if len(X_test) == 0:
    raise ValueError("No valid test samples found.")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=LABELS))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
