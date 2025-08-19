import cv2

# Haar cascades (XML files must be in the same folder)
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Could not load Haar cascade XML files. Make sure they're in the same folder.")

# Load a single image (change filename)
image_path = "data/drowsy/A0005.png"
image = cv2.imread(image_path)

if image is None:
    raise IOError(f"Could not read image: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(25, 10),
        maxSize=(80, 60)
    )

    # Keep only eyes in upper half of face
    filtered_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey + eh / 2 < h / 2]

    for (ex, ey, ew, eh) in filtered_eyes:
        print("Ratio:", eh/ew)
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Show result
cv2.imshow("Haar Eye Detection (Image)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
