import cv2
import dlib
import time
import numpy as np
import pickle
import threading
from imutils import face_utils
from playsound import playsound

MODEL_TYPE = "svc"

if MODEL_TYPE == "svc":
    model_path = "models/svc_ear.pkl"
elif MODEL_TYPE == "logreg":
    model_path = "models/drowsiness_model.pkl"

alert_playing = False

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def play_alert_sound():
    global alert_playing
    alert_playing = True
    playsound("alert3.mp3")
    alert_playing = False

with open(model_path, "rb") as f:
    model = pickle.load(f)

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(1)

drowsy_start_time = None
DROWSY_THRESHOLD_SEC = 2.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    ear = None
    pred = None
    prob = None

    if len(rects) > 0:
        rect = max(rects, key=lambda r: r.width() * r.height())

        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        leftEye = shape_np[42:48]
        rightEye = shape_np[36:42]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        pred = model.predict([[ear]])[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([[ear]])[0][pred]
        else:
            prob = None

        if pred == 1:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
            elapsed = time.time() - drowsy_start_time
            if elapsed >= DROWSY_THRESHOLD_SEC:
                text_size = cv2.getTextSize("DROWSINESS ALERT!", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                x_pos = frame.shape[1] - text_size[0] - 50
                cv2.putText(frame, "DROWSINESS ALERT!", (x_pos, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if not alert_playing:
                    threading.Thread(target=play_alert_sound, daemon=True).start()
        else:
            drowsy_start_time = None

        cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    overlay = frame.copy()
    stats_h = 90
    cv2.rectangle(overlay, (10, 10), (300, 10 + stats_h), (50, 50, 50), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if prob is not None:
        cv2.putText(frame, f"Prob: {prob:.2f}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if pred is not None:
        status_text = "Closed" if pred == 1 else "Open"
        cv2.putText(frame, f"Status: {status_text}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
