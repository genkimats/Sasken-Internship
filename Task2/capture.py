import cv2
import os
import time

def continuous_capture(output_dir="blink_dataset/close", filename_prefix="tokunaga"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open camera (device index 2 by default)
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    print("Press 'q' to stop capturing...")
    for i in range(1, 501):
        print(i)
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Generate filename using timestamp in milliseconds
        filename = f"{filename_prefix}_{i}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Save the frame efficiently
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Show live preview (optional, remove for faster performance)
        cv2.imshow("Camera", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped capturing.")

if __name__ == "__main__":
    continuous_capture()
