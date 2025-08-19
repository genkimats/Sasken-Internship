import os
import cv2
import dlib
import imutils
import matplotlib.pyplot as plt
from scipy.spatial import distance
from imutils import face_utils

# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Initialize detector and predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Directories
base_dirs = {
    "Drowsy": "data/drowsy",          # folder with drowsy images
    "Non-Drowsy": "data/non_drowsy"   # folder with non-drowsy images
}

ear_data = {label: [] for label in base_dirs}

# Process each folder
for label, dir_path in base_dirs.items():
    n = 0
    for filename in sorted(os.listdir(dir_path)):
        n += 1
        if n > 2000:
          break
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(dir_path, filename)
            print(f"[{label}] Processing: {img_path}")  # log the file being processed
            
            frame = cv2.imread(img_path)
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            subjects = detect(gray, 0)
            if len(subjects) == 0:
                print(f"  ⚠ No face detected in {filename}")
                continue

            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                ear_data[label].append(ear)

# Plot
plt.figure(figsize=(10, 5))
for label, values in ear_data.items():
    plt.plot(range(len(values)), values, marker='o', label=label)

# Optional threshold line
thresh = 0.21
plt.axhline(y=thresh, color='r', linestyle='--', label="Threshold")

plt.title("EAR Values: Drowsy vs Non-Drowsy")
plt.xlabel("Image Index")
plt.ylabel("EAR")
plt.legend()
plt.grid(True)
plt.show()
