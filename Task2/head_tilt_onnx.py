import cv2
import numpy as np
import onnxruntime as ort

# ===== Thresholds =====
PITCH_THRESHOLD = 20
ROLL_THRESHOLD = 15

# ===== Load ONNX Models =====
pnet_session = ort.InferenceSession("pnet.onnx")
rnet_session = ort.InferenceSession("rnet.onnx")
onet_session = ort.InferenceSession("onet.onnx")

# ===== Landmark Indices =====
left_eye_indices = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173]
right_eye_indices = [362, 263, 387, 386, 385, 373, 380, 381, 382, 362]
outer_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# ===== Functions =====
def get_landmark_center(landmarks, indices):
    pts = np.array([landmarks[i] for i in indices])
    return np.mean(pts, axis=0)

def compute_pitch_roll(landmarks):
    # Get 5 landmarks
    left_eye = get_landmark_center(landmarks, left_eye_indices)
    right_eye = get_landmark_center(landmarks, right_eye_indices)
    outer_lip_center = get_landmark_center(landmarks, outer_lip_indices)
    nose_tip = landmarks[1]  # Usually index 1
    chin = landmarks[152]    # Chin landmark

    # Roll angle (head tilt)
    eye_line = right_eye - left_eye
    roll = np.degrees(np.arctan2(eye_line[1], eye_line[0]))

    # Pitch angle (looking up/down)
    nose_to_lips = outer_lip_center - nose_tip
    pitch = np.degrees(np.arctan2(nose_to_lips[1], np.linalg.norm(eye_line)))

    return pitch, roll

def draw_angles(frame, pitch, roll):
    # Color changes if exceeding thresholds
    pitch_color = (0, 255, 0) if abs(pitch) <= PITCH_THRESHOLD else (0, 0, 255)
    roll_color = (0, 255, 0) if abs(roll) <= ROLL_THRESHOLD else (0, 0, 255)

    cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, pitch_color, 2)
    cv2.putText(frame, f"Roll:  {roll:.1f} deg", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, roll_color, 2)

    # If exceeding threshold, display alert message
    if abs(pitch) > PITCH_THRESHOLD:
        cv2.putText(frame, "PITCH ALERT!", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    if abs(roll) > ROLL_THRESHOLD:
        cv2.putText(frame, "ROLL ALERT!", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

# ===== Video Capture =====
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # TODO: Replace this part with your existing ONNX inference
    # landmarks = your_onnx_landmark_detection(frame)
    landmarks = []  # <-- placeholder

    if len(landmarks) > 0:
        pitch, roll = compute_pitch_roll(landmarks)
        draw_angles(frame, pitch, roll)

    cv2.imshow("Face Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
