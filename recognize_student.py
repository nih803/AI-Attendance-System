
import cv2
from pathlib import Path
import numpy as np

from src.attendance.database import mark_attendance

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
ENROLLMENT_DIR = PROJECT_ROOT / "data" / "enrollment"
MODEL_DIR = PROJECT_ROOT / "models"

# Recognition threshold: demo value only; calibrate before real use.
THRESHOLD = 0.40
SESSION = "Morning"

# Load the pretrained models
detector = cv2.FaceDetectorYN_create(
    str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"),
    "",
    (320, 320)
)

recognizer = cv2.FaceRecognizerSF_create(
    str(MODEL_DIR / "face_recognition_sface_2021dec.onnx"),
    ""
)


def load_student_references():
    """Build one reference facial feature vector per enrolled student."""
    references = {}

    student_folders = [
        folder for folder in ENROLLMENT_DIR.iterdir()
        if folder.is_dir()
    ] if ENROLLMENT_DIR.exists() else []

    for folder in student_folders:
        student_id = folder.name
        features_list = []

        for image_path in folder.glob("*.jpg"):
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Could not read {image_path.name}")
                continue

            height, width = image.shape[:2]
            detector.setInputSize((width, height))
            _, faces = detector.detect(image)

            if faces is None or len(faces) != 1:
                print(f"Skipping unsuitable image: {image_path.name}")
                continue

            aligned_face = recognizer.alignCrop(image, faces[0])
            features = recognizer.feature(aligned_face).flatten()
            norm = np.linalg.norm(features)

            if norm > 0:
                features_list.append(features / norm)

        if features_list:
            reference = np.mean(features_list, axis=0)
            norm = np.linalg.norm(reference)

            if norm > 0:
                references[student_id] = reference / norm
                print(
                    f"Loaded {len(features_list)} reference images "
                    f"for {student_id}."
                )
        else:
            print(f"No usable reference images for {student_id}.")

    return references


student_references = load_student_references()

if not student_references:
    raise SystemExit(
        "No usable student references found. Check data/enrollment."
    )

print(f"\nLoaded {len(student_references)} enrolled student(s).")
print("Look at the webcam. Press Q to quit.")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open webcam.")

# Remember successful attendance attempts during this run.
attendance_marked = set()

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)

        label = "No face detected"
        color = (0, 0, 255)

        if faces is not None:
            if len(faces) != 1:
                label = "Show only one face"
            else:
                face = faces[0]
                aligned_face = recognizer.alignCrop(frame, face)
                features = recognizer.feature(aligned_face).flatten()
                norm = np.linalg.norm(features)

                if norm > 0:
                    features = features / norm

                    # Compare this face with every enrolled student.
                    scores = {
                        sid: float(np.dot(reference, features))
                        for sid, reference in student_references.items()
                    }

                    ranked = sorted(
                        scores.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )
                    best_id, best_score = ranked[0]

                    # Require a clear margin over the second-best match.
                    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
                    margin = best_score - second_score

                    x, y, w, h = face[:4].astype(int)

                    if best_score >= THRESHOLD and (
                        len(ranked) == 1 or margin >= 0.05
                    ):
                        label = f"{best_id} | Similarity: {best_score:.2f}"
                        color = (0, 255, 0)

                        if best_id not in attendance_marked:
                            mark_attendance(best_id, SESSION)
                            attendance_marked.add(best_id)
                    elif best_score >= THRESHOLD:
                        label = "Ambiguous match - not marked"
                        color = (0, 165, 255)
                    else:
                        label = f"Unknown | Similarity: {best_score:.2f}"
                        color = (0, 0, 255)

                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), color, 2
                    )

        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.imshow("Smart Student Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
