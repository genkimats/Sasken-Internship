import cv2
import dlib
import numpy as np
import os
import csv
from imutils import face_utils

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

dataset_path = "data"
output_csv = "ear_dataset.csv"

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["ear", "label"])

    for label in ["drowsy", "non_drowsy"]:
        folder = os.path.join(dataset_path, label)
        for file_name in os.listdir(folder):
            img_path = os.path.join(folder, file_name)
            print(f"[INFO] Processing {img_path} ...")

            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Could not read {img_path}, skipping.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            if len(rects) == 0:
                print(f"[WARNING] No face detected in {img_path}, skipping.")
                continue

            for rect in rects:
                shape = predictor(gray, rect)
                shape_np = face_utils.shape_to_np(shape)

                leftEye = shape_np[42:48]
                rightEye = shape_np[36:42]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                writer.writerow([ear, 1 if label == "drowsy" else 0])

print(f"[INFO] EAR data saved to {output_csv}")
