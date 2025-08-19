# detect_drowsiness_svr.py
import cv2
import dlib
import numpy as np
import pickle
import time

# Paths
haar_path = "haarcascade/haarcascade_eye.xml"
model_path = "models/ear_predictor_svr.pkl"

# Load model
with open(model_path, "rb") as f:
    svr = pickle.load(f)

# Haar cascade
eye_cascade = cv2.CascadeClassifier(haar_path)

# Drowsiness parameters
EAR_THRESHOLD = 0.25
CLOSED_TIME_THRESHOLD = 2.0
drowsy_start = None

cap = cv2.VideoCapture(2)  # default device 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (ex, ey, ew, eh) in eyes:
        eye_img = gray[ey:ey+eh, ex:ex+ew]
        eye_img = cv2.resize(eye_img, (24, 24))
        eye_flat = eye_img.flatten().reshape(1, -1)

        predicted_ear = svr.predict(eye_flat)[0]
        cv2.putText(frame, f"EAR: {predicted_ear:.2f}", (ex, ey-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if predicted_ear < EAR_THRESHOLD:
            if drowsy_start is None:
                drowsy_start = time.time()
            elif time.time() - drowsy_start > CLOSED_TIME_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        else:
            drowsy_start = None

        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)

    cv2.imshow("Drowsiness Detection (SVR EAR Prediction)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
