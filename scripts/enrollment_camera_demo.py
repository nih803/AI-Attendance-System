
import cv2

from src.attendance.recognition import load_models
from src.attendance.enrollment import save_face_image


# Load face recognition models
detector, recognizer = load_models()

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    exit()

print("Press SPACE to capture your face.")
print("Press Q to quit.")

captured_frame = None

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read webcam.")
        break

    cv2.imshow("Enrollment Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        captured_frame = frame.copy()
        break

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()


# Save captured image
if captured_frame is not None:

    class UploadedImage:
        def getvalue(self):
            success, encoded = cv2.imencode(
                ".jpg",
                captured_frame
            )
            if not success:
                raise ValueError("Image encoding failed.")
            return encoded.tobytes()

    uploaded_image = UploadedImage()

    student_id = input("Enter a NEW test student ID: ").strip()
    if not student_id:
        raise ValueError("Student ID cannot be empty.")

    success, message = save_face_image(
        student_id,
        uploaded_image,
        detector,
        recognizer
    )

    print(message)

else:
    print("No image captured.")
