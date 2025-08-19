import cv2
import joblib

# Load Haar cascades
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")

# Load SVM model
svm_model = joblib.load("models/haar_ear_svm.pkl")

# Parameters
TIME_THRESHOLD_SEC = 2.0
FPS = 20
FRAME_THRESHOLD = int(TIME_THRESHOLD_SEC * FPS)
closed_frames = 0
alarm_on = False

cap = cv2.VideoCapture(2)

print("[INFO] Starting Driver Drowsiness Detection...")
print(f"[INFO] Time threshold: {TIME_THRESHOLD_SEC} sec ({FRAME_THRESHOLD} frames)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(f"[DEBUG] Faces detected: {len(faces)}")
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(
            roi_gray, 1.1, 10, minSize=(25, 25), maxSize=(80, 80)
        )
        print(f"[DEBUG] Raw eyes detected: {len(eyes)}")
        
        # Filter: keep only eyes in upper half of face
        filtered_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey + eh/2 < h/2]
        print(f"[DEBUG] Filtered eyes (upper half): {len(filtered_eyes)}")
        
        eye_closed_count = 0
        
        for i, (ex, ey, ew, eh) in enumerate(filtered_eyes, start=1):
            aspect_ratio = eh / ew
            pred = svm_model.predict([[aspect_ratio]])[0]  # 0=closed, 1=open
            
            print(f"[LOG] Eye {i}: EAR={aspect_ratio:.3f}, Prediction={'OPEN' if pred==1 else 'CLOSED'}")
            
            if pred == 0:
                eye_closed_count += 1
            
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
        
        # Frame-based drowsiness check
        if eye_closed_count == len(filtered_eyes) and len(filtered_eyes) > 0:
            closed_frames += 1
            print(f"[DEBUG] Closed frames count: {closed_frames}/{FRAME_THRESHOLD}")
        else:
            if closed_frames > 0:
                print("[DEBUG] Reset closed frames counter.")
            closed_frames = 0
            alarm_on = False
        
        if closed_frames >= FRAME_THRESHOLD:
            if not alarm_on:
                print("[ALERT] Drowsiness detected! Eyes closed for threshold time.")
            alarm_on = True
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("Haar EAR + SVM", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        print("[INFO] ESC pressed. Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
