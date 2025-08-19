import cv2
import dlib
import numpy as np
import pickle
from scipy.spatial import distance as dist

# ===== Load trained SVC model =====
with open("models/svc_ear.pkl", "rb") as f:
    model = pickle.load(f)

# ===== Dlib face detector & landmark predictor =====
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ===== Eye landmark indexes (68-point model) =====
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

def eye_aspect_ratio(eye):
    # eye: array of 6 (x, y) coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ===== Drowsiness parameters =====
CONSEC_FRAMES = 15
frame_counter = 0
drowsy_alert = False

# ===== Start Webcam =====
cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        leftEye = shape_np[LEFT_EYE]
        rightEye = shape_np[RIGHT_EYE]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Predict with trained SVC
        pred = model.predict([[ear]])[0]
        print(pred)

        if pred == 1:  # Assuming 1 = drowsy
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
                drowsy_alert = True
        else:
            frame_counter = 0
            drowsy_alert = False

        # Draw eye landmarks
        for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display EAR and status
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status_text = "DROWSY" if drowsy_alert else "Awake"
        color = (0, 0, 255) if drowsy_alert else (0, 255, 0)
        cv2.putText(frame, status_text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
