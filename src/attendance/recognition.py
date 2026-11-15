"""Face matching helpers used by the AttendAI dashboard."""

from pathlib import Path
import logging

import cv2
import numpy as np

from src.attendance.database import list_students, mark_attendance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENROLLMENT_DIR = PROJECT_ROOT / "data" / "enrollment"
MODEL_DIR = PROJECT_ROOT / "models"

# These are prototype values. Evaluate them on consented, representative data
# before relying on the system for real attendance decisions.
THRESHOLD = 0.40
MARGIN_THRESHOLD = 0.05
SESSION = "Morning"
logger = logging.getLogger(__name__)

_REFERENCE_LOAD_DIAGNOSTICS = {
    "loaded_references": 0,
    "loaded_images": 0,
    "skipped_images": 0,
    "invalid_features": 0,
    "ignored_folders": 0,
    "students_without_references": 0,
}


def get_reference_load_diagnostics():
    """Return aggregate reference-loading counts without identifying records."""
    return dict(_REFERENCE_LOAD_DIAGNOSTICS)


def load_models():
    """Load OpenCV Zoo YuNet detector and SFace recognizer models."""
    detector_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    recognizer_path = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
    for model_path, label in (
        (detector_path, "Face detection"),
        (recognizer_path, "Face recognition"),
    ):
        if not model_path.is_file():
            raise FileNotFoundError(f"{label} model is missing: {model_path}")

    try:
        detector = cv2.FaceDetectorYN_create(
            str(detector_path), "", (320, 320)
        )
        recognizer = cv2.FaceRecognizerSF_create(str(recognizer_path), "")
    except cv2.error as error:
        raise RuntimeError("OpenCV could not load the face models") from error
    return detector, recognizer


def load_student_references(detector, recognizer, registered_student_ids=None):
    """Build validated normalized references for currently registered students.

    Diagnostics contain counts only; no IDs, paths, images, or features are logged.
    """
    references = {}
    if registered_student_ids is None:
        registered_student_ids = (student_id for student_id, _ in list_students())
    registered_ids = {
        student_id for student_id in registered_student_ids
        if isinstance(student_id, str) and student_id
    }
    diagnostics = {
        "loaded_references": 0,
        "loaded_images": 0,
        "skipped_images": 0,
        "invalid_features": 0,
        "ignored_folders": 0,
        "students_without_references": 0,
    }

    if ENROLLMENT_DIR.is_dir():
        for folder in sorted(ENROLLMENT_DIR.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name not in registered_ids:
                diagnostics["ignored_folders"] += 1
                continue

            feature_vectors = []
            expected_shape = None
            image_paths = sorted(folder.glob("*.jpg"))
            for image_path in image_paths:
                try:
                    image = cv2.imread(str(image_path))
                except cv2.error:
                    image = None
                if (
                    image is None
                    or not isinstance(image, np.ndarray)
                    or image.size == 0
                    or image.ndim != 3
                    or image.shape[2] != 3
                ):
                    diagnostics["skipped_images"] += 1
                    continue

                height, width = image.shape[:2]
                if height <= 0 or width <= 0 or height * width > 20_000_000:
                    diagnostics["skipped_images"] += 1
                    continue
                try:
                    detector.setInputSize((width, height))
                    _, faces = detector.detect(image)
                    if faces is None or len(faces) != 1:
                        diagnostics["skipped_images"] += 1
                        continue
                    aligned_face = recognizer.alignCrop(image, faces[0])
                    feature = np.asarray(
                        recognizer.feature(aligned_face), dtype=np.float32
                    ).reshape(-1)
                except (cv2.error, ValueError, TypeError):
                    diagnostics["skipped_images"] += 1
                    continue

                norm = np.linalg.norm(feature)
                if (
                    feature.size == 0
                    or not np.isfinite(norm)
                    or norm <= 0
                    or not np.all(np.isfinite(feature))
                    or (expected_shape is not None and feature.shape != expected_shape)
                ):
                    diagnostics["invalid_features"] += 1
                    continue

                expected_shape = feature.shape
                feature_vectors.append(feature / norm)
                diagnostics["loaded_images"] += 1

            if feature_vectors:
                reference = np.mean(feature_vectors, axis=0)
                norm = np.linalg.norm(reference)
                if norm > 0 and np.isfinite(norm) and np.all(np.isfinite(reference)):
                    references[folder.name] = reference / norm
                else:
                    diagnostics["invalid_features"] += len(feature_vectors)

    diagnostics["loaded_references"] = len(references)
    diagnostics["students_without_references"] = len(registered_ids - references.keys())
    _REFERENCE_LOAD_DIAGNOSTICS.update(diagnostics)
    logger.info(
        "Enrollment reference load: %d references loaded from %d images; "
        "%d images skipped; %d invalid features; %d stale folders ignored; "
        "%d registered students without a usable reference.",
        diagnostics["loaded_references"],
        diagnostics["loaded_images"],
        diagnostics["skipped_images"],
        diagnostics["invalid_features"],
        diagnostics["ignored_folders"],
        diagnostics["students_without_references"],
    )
    return references


def _recognition_result(status, message, student_id=None, similarity=None):
    return {
        "status": status,
        "student_id": student_id,
        "similarity": similarity,
        "message": message,
    }


def recognize_face(frame, detector, recognizer, student_references):
    """Match exactly one face; uncertain or malformed input never passes."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return _recognition_result("error", "Invalid webcam frame")
    if not student_references:
        return _recognition_result("error", "No enrolled students available")

    height, width = frame.shape[:2]
    try:
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            return _recognition_result("no_face", "No face detected")
        if len(faces) > 1:
            return _recognition_result("multiple_faces", "Show only one face")

        aligned_face = recognizer.alignCrop(frame, faces[0])
        feature = np.asarray(
            recognizer.feature(aligned_face), dtype=np.float32
        ).reshape(-1)
    except (cv2.error, ValueError, TypeError):
        return _recognition_result("error", "Could not process the face image")

    norm = np.linalg.norm(feature)
    if norm == 0 or not np.isfinite(norm) or not np.all(np.isfinite(feature)):
        return _recognition_result("error", "Could not extract facial features")
    feature /= norm

    scores = {}
    for student_id, reference in student_references.items():
        reference = np.asarray(reference, dtype=np.float32).reshape(-1)
        if reference.shape == feature.shape and np.all(np.isfinite(reference)):
            scores[student_id] = float(np.dot(reference, feature))
    if not scores:
        return _recognition_result("error", "No valid face references available")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - second_score

    if best_score < THRESHOLD:
        return _recognition_result(
            "unknown", f"Unknown face - Similarity: {best_score:.2f}",
            similarity=best_score,
        )
    if len(ranked) > 1 and margin < MARGIN_THRESHOLD:
        return _recognition_result(
            "ambiguous", "Ambiguous match - not marked", similarity=best_score
        )
    return _recognition_result(
        "recognized", f"Recognized: {best_id}", best_id, best_score
    )


def process_recognition_result(result, attendance_marked):
    """Persist only recognized students and report database outcomes accurately."""
    if not isinstance(result, dict):
        return {"status": "error", "message": "Invalid recognition result"}
    if result.get("status") != "recognized":
        return {
            "status": result.get("status", "error"),
            "message": result.get("message", "Recognition did not complete"),
        }

    student_id = result.get("student_id")
    if not isinstance(student_id, str) or not student_id:
        return {"status": "error", "message": "Recognized student ID is missing"}
    if student_id in attendance_marked:
        return {
            "status": "already_processed",
            "student_id": student_id,
            "message": f"Attendance already processed for {student_id}",
        }

    try:
        database_status = mark_attendance(student_id, SESSION)
    except Exception:
        return {
            "status": "error",
            "student_id": student_id,
            "message": "Could not save attendance. Please try again.",
        }

    if database_status == "success":
        attendance_marked.add(student_id)
        return {
            "status": "attendance_processed",
            "student_id": student_id,
            "message": f"Attendance processed for {student_id}",
        }
    if database_status == "duplicate":
        attendance_marked.add(student_id)
        return {
            "status": "already_processed",
            "student_id": student_id,
            "message": f"Attendance was already recorded for {student_id}",
        }
    return {
        "status": "error",
        "student_id": student_id,
        "message": "Student is not registered; attendance was not recorded.",
    }
