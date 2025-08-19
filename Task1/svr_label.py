# label_ear_dataset.py
import cv2
import dlib
import os
import csv
import numpy as np
from imutils import face_utils

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

predictor_path = "shape_predictor_68_face_landmarks.dat"
dataset_dirs = ["data/drowsy", "data/non_drowsy"]
output_csv = "ear_labels.csv"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filepath", "ear"])

    for dataset_dir in dataset_dirs:
        for filename in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, filename)
            print(f"[INFO] Processing image: {path}")

            img = cv2.imread(path)
            if img is None:
                print(f"[WARNING] Could not read {path}, skipping.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            if len(rects) == 0:
                print(f"[WARNING] No face detected in {path}, skipping.")
                continue

            for rect in rects:
                shape = predictor(gray, rect)
                shape_np = face_utils.shape_to_np(shape)
                leftEye = shape_np[42:48]
                rightEye = shape_np[36:42]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                print(f"[DEBUG] EAR for {path}: {ear:.4f}")
                writer.writerow([path, ear])

print(f"[INFO] EAR labels saved to {output_csv}")
