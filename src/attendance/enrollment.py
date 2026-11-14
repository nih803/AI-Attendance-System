"""Validation and local storage for enrolled student face images."""

import re
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENROLLMENT_DIR = PROJECT_ROOT / "data" / "enrollment"
MAX_IMAGE_SIZE = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def validate_student_id(student_id):
    """Allow simple IDs only, so they cannot escape the enrollment folder."""
    return bool(
        isinstance(student_id, str)
        and len(student_id) <= 30
        and re.fullmatch(r"[A-Za-z0-9_-]+", student_id.strip())
    )


def save_face_image(student_id, uploaded_image, detector, recognizer):
    """Validate and store one aligned face image in the local data folder."""
    if not validate_student_id(student_id):
        return False, "Invalid student ID."
    if uploaded_image is None or not hasattr(uploaded_image, "getvalue"):
        return False, "No image was provided."

    try:
        image_data = uploaded_image.getvalue()
    except (OSError, ValueError):
        return False, "Could not read the uploaded image. Please try again."
    if not image_data:
        return False, "The uploaded image is empty."
    if len(image_data) > MAX_IMAGE_SIZE:
        return False, "Image is too large. Maximum allowed size is 5 MB."

    try:
        frame = cv2.imdecode(
            np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
    except cv2.error:
        frame = None
    if frame is None:
        return False, "Could not read the uploaded image."

    height, width = frame.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        return False, "Image dimensions are too large."
    if width < 100 or height < 100:
        return False, "Image resolution is too small."

    try:
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            return False, "No face detected. Try again."
        if len(faces) > 1:
            return False, "Multiple faces detected. Show only one face."
        aligned_face = recognizer.alignCrop(frame, faces[0])
        success, encoded_image = cv2.imencode(".jpg", aligned_face)
        if not success:
            return False, "Could not prepare the face image for storage."
    except cv2.error:
        return False, "Could not process the face image. Please try again."

    save_folder = ENROLLMENT_DIR / student_id.strip()
    try:
        save_folder.mkdir(parents=True, exist_ok=True)
        count = 1
        while (save_folder / f"{count:03d}.jpg").exists():
            count += 1
        image_path = save_folder / f"{count:03d}.jpg"
        image_path.write_bytes(encoded_image.tobytes())
    except OSError:
        return False, "Could not save the face image to local storage."

    return True, "Face image saved successfully."
