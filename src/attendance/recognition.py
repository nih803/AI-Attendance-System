
import cv2
from pathlib import Path
import numpy as np


# ==========================================
# PROJECT PATHS AND SETTINGS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENROLLMENT_DIR = PROJECT_ROOT / "data" / "enrollment"
MODEL_DIR = PROJECT_ROOT / "models"

THRESHOLD = 0.40
MARGIN_THRESHOLD = 0.05

SESSION = "Morning"


# ==========================================
# LOAD PRETRAINED MODELS
# ==========================================

def load_models():
    """
    Load YuNet face detector and SFace face recognizer.
    """

    detector_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    recognizer_path = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

    if not detector_path.exists():
        raise FileNotFoundError(
            f"Face detection model not found: {detector_path}"
        )

    if not recognizer_path.exists():
        raise FileNotFoundError(
            f"Face recognition model not found: {recognizer_path}"
        )

    detector = cv2.FaceDetectorYN_create(
        str(detector_path),
        "",
        (320, 320)
    )

    recognizer = cv2.FaceRecognizerSF_create(
        str(recognizer_path),
        ""
    )

    return detector, recognizer


# ==========================================
# LOAD ENROLLED STUDENT REFERENCES
# ==========================================

def load_student_references(detector, recognizer):
    """
    Generate one reference facial feature vector
    for every enrolled student.
    """

    references = {}

    if not ENROLLMENT_DIR.exists():
        print("Enrollment directory does not exist.")
        return references

    student_folders = [
        folder for folder in ENROLLMENT_DIR.iterdir()
        if folder.is_dir()
    ]

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

            aligned_face = recognizer.alignCrop(
                image,
                faces[0]
            )

            features = recognizer.feature(
                aligned_face
            ).flatten()

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


# ==========================================
# RECOGNIZE FACE FROM ONE FRAME
# ==========================================

def recognize_face(frame, detector, recognizer, student_references):
    """
    Recognize a face in one webcam frame.

    Returns a dictionary containing:
    status, student_id, similarity and message.
    """

    # Check whether frame is valid
    if frame is None or frame.size == 0:
        return {
            "status": "error",
            "student_id": None,
            "similarity": None,
            "message": "Invalid webcam frame"
        }

    if not student_references:
        return {
            "status": "error",
            "student_id": None,
            "similarity": None,
            "message": "No enrolled students available"
        }

    # Detect faces
    height, width = frame.shape[:2]

    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    # No face detected
    if faces is None or len(faces) == 0:
        return {
            "status": "no_face",
            "student_id": None,
            "similarity": None,
            "message": "No face detected"
        }

    # Multiple faces detected
    if len(faces) > 1:
        return {
            "status": "multiple_faces",
            "student_id": None,
            "similarity": None,
            "message": "Show only one face"
        }

    # Extract facial features
    face = faces[0]

    aligned_face = recognizer.alignCrop(
        frame,
        face
    )

    features = recognizer.feature(
        aligned_face
    ).flatten()

    norm = np.linalg.norm(features)

    if norm == 0:
        return {
            "status": "error",
            "student_id": None,
            "similarity": None,
            "message": "Could not extract facial features"
        }

    features = features / norm

    # Compare against enrolled students
    scores = {
        student_id: float(np.dot(reference, features))
        for student_id, reference in student_references.items()
    }

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_id, best_score = ranked[0]

    # Calculate difference between best and second-best match
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0

    margin = best_score - second_score

    # Check recognition threshold
    if best_score >= THRESHOLD:

        if len(ranked) == 1 or margin >= MARGIN_THRESHOLD:

            return {
                "status": "recognized",
                "student_id": best_id,
                "similarity": best_score,
                "message": f"Recognized: {best_id}"
            }

        return {
            "status": "ambiguous",
            "student_id": None,
            "similarity": best_score,
            "message": "Ambiguous match - not marked"
        }

    # Unknown face
    return {
        "status": "unknown",
        "student_id": None,
        "similarity": best_score,
        "message": f"Unknown face - Similarity: {best_score:.2f}"
    }


from src.attendance.database import mark_attendance


# ==========================================
# PROCESS RECOGNITION RESULT
# ==========================================

def process_recognition_result(result, attendance_marked):
    """
    Process a recognition result and mark attendance
    only for successfully recognized students.
    """

    if result["status"] != "recognized":
        return {
            "status": result["status"],
            "message": result["message"]
        }

    student_id = result["student_id"]

    # Prevent repeated attempts during the current session
    if student_id in attendance_marked:
        return {
            "status": "already_processed",
            "student_id": student_id,
            "message": f"Attendance already processed for {student_id}"
        }

    # Record attendance in SQLite
    mark_attendance(student_id, SESSION)

    attendance_marked.add(student_id)

    return {
        "status": "attendance_processed",
        "student_id": student_id,
        "message": f"Attendance processed for {student_id}"
    }
