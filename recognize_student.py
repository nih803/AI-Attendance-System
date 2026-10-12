
import cv2
from pathlib import Path
import numpy as np

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

# Find the enrolled student's images
student_id = input("Enter enrolled student ID (ST001): ").strip()
student_folder = Path("data") / "enrollment" / student_id
image_paths = list(student_folder.glob("*.jpg"))

if not image_paths:
    raise SystemExit("No enrollment images found for this student.")

# Generate face representations from enrollment images
reference_features = []

for image_path in image_paths:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path}")
        continue

    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)

    if faces is None or len(faces) != 1:
        print(f"Skipping unsuitable image: {image_path.name}")
        continue

    aligned_face = recognizer.alignCrop(image, faces[0])
    features = recognizer.feature(aligned_face)
    reference_features.append(features.flatten())

if not reference_features:
    raise SystemExit("Could not extract features from enrollment images.")

# Average the reference representations
reference = np.mean(reference_features, axis=0)
reference = reference / np.linalg.norm(reference)

print(f"Loaded {len(reference_features)} reference images.")
print("Look at the webcam. Press Q to quit.")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open webcam.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    label = "No face detected"

    if faces is not None:
        if len(faces) == 1:
            face = faces[0]
            aligned_face = recognizer.alignCrop(frame, face)
            features = recognizer.feature(aligned_face).flatten()
            features = features / np.linalg.norm(features)

            # Cosine similarity: higher means more similar
            similarity = float(np.dot(reference, features))

            x, y, w, h = face[:4].astype(int)

            # Initial demo threshold; must be calibrated
            if similarity >= 0.40:
                label = f"{student_id} | Similarity: {similarity:.2f}"
                color = (0, 255, 0)
            else:
                label = f"Unknown | Similarity: {similarity:.2f}"
                color = (0, 0, 255)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), color, 2
            )
        else:
            label = "Show only one face"

    cv2.putText(
        frame,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.imshow("Student Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
