
import cv2
from pathlib import Path

# Load the pretrained models
detector = cv2.FaceDetectorYN_create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320)
)

recognizer = cv2.FaceRecognizerSF_create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

# Create a folder for the test student's images
student_id = input("Enter test student ID (example ST001): ").strip()

if not student_id or not all(
    c.isalnum() or c in "_-" for c in student_id
):
    raise ValueError("Use letters, numbers, underscores or hyphens only.")

save_folder = Path("data") / "enrollment" / student_id
save_folder.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open webcam.")

count = len(list(save_folder.glob("*.jpg")))

print("Press S to save a face image.")
print("Change your head angle slightly between captures.")
print("Press Q to finish.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

    cv2.putText(
        frame,
        f"Saved: {count} | Press S to capture",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.imshow("Student Enrollment", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if faces is None or len(faces) != 1:
            print("Ensure exactly one face is visible.")
            continue

        # Align the detected face for recognition
        aligned_face = recognizer.alignCrop(frame, faces[0])

        count += 1
        image_path = save_folder / f"{count:03d}.jpg"

        if cv2.imwrite(str(image_path), aligned_face):
            print(f"Saved: {image_path}")
        else:
            count -= 1
            print("Image could not be saved.")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"Enrollment finished. Images saved: {count}")
