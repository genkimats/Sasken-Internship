import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Neutral pitch offset (default resting head tilt in degrees)
NEUTRAL_PITCH = 23.0

# Thresholds for roll (left/right tilt)
ROLL_LEFT_THRESHOLD = -15    # Negative roll → tilt left
ROLL_RIGHT_THRESHOLD = 15    # Positive roll → tilt right

# Thresholds for pitch (forward/backward tilt)
PITCH_FORWARD_THRESHOLD = 5     # Positive pitch → lean forward
PITCH_BACKWARD_THRESHOLD = -15   # Negative pitch → lean backward

# Start webcam
cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # === Get important landmarks ===
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]

            # Convert to pixel coordinates
            left_eye = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
            right_eye = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))
            nose = (int(nose_tip.x * w), int(nose_tip.y * h))

            # Draw points
            cv2.circle(frame, left_eye, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye, 3, (0, 255, 0), -1)
            cv2.circle(frame, nose, 3, (0, 0, 255), -1)

            # Draw line between eyes
            cv2.line(frame, left_eye, right_eye, (255, 0, 0), 2)

            # === LEFT/RIGHT TILT (ROLL) ===
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            roll_angle = math.degrees(math.atan2(dy, dx))

            # === FRONT/BACKWARD TILT (PITCH) ===
            mid_x = (left_eye[0] + right_eye[0]) // 2
            mid_y = (left_eye[1] + right_eye[1]) // 2
            midpoint = (mid_x, mid_y)

            cv2.circle(frame, midpoint, 3, (255, 255, 0), -1)
            cv2.line(frame, midpoint, nose, (0, 255, 255), 2)

            vertical_diff = nose[1] - midpoint[1]
            eye_distance = math.dist(left_eye, right_eye)
            raw_pitch = math.degrees(math.atan2(vertical_diff, eye_distance))

            # Adjust pitch by subtracting neutral offset
            pitch_angle = raw_pitch - NEUTRAL_PITCH

            # Display angles
            cv2.putText(frame, f"Roll: {roll_angle:.2f} deg", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch_angle:.2f} deg", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # === WARNINGS ===
            warning = None

            # Check roll thresholds
            if roll_angle > ROLL_RIGHT_THRESHOLD:
                warning = "TILT RIGHT TOO MUCH!"
            elif roll_angle < ROLL_LEFT_THRESHOLD:
                warning = "TILT LEFT TOO MUCH!"

            # Check pitch thresholds
            elif pitch_angle > PITCH_FORWARD_THRESHOLD:
                warning = "LEANING FORWARD TOO MUCH!"
            elif pitch_angle < PITCH_BACKWARD_THRESHOLD:
                warning = "LEANING BACKWARD TOO MUCH!"

            # Display warning if needed
            if warning:
                cv2.putText(frame, warning, (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Draw landmarks for visualization
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # Show frame
    cv2.imshow("Head Tilt Detection (Yaw + Pitch)", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
