import cv2

# Haar cascades (XML files must be in the same folder as this script)
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Could not load Haar cascade XML files. Make sure they're in the same folder.")

# Open camera (device 2 by default)
cap = cv2.VideoCapture(2)

# Drowsiness detection parameters
TIME_THRESHOLD_SEC = 2.0
FPS = 20  # estimated
FRAME_THRESHOLD = int(TIME_THRESHOLD_SEC * FPS)
closed_frames = 0
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(25, 10),
            maxSize=(90, 50)
        )

        # Filter out non-eye regions (only keep eyes in top half of face)
        filtered_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey + eh / 2 < h / 2]

        if len(filtered_eyes) == 0:
            closed_frames += 1
        else:
            closed_frames = 0
            alarm_on = False

        for (ex, ey, ew, eh) in filtered_eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Drowsiness alert
        if closed_frames >= FRAME_THRESHOLD:
            if not alarm_on:
                print("[ALERT] Drowsiness detected!")
            alarm_on = True
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Haar Eye Detection (Filtered)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
