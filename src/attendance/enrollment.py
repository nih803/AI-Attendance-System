
import cv2
import numpy as np

from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENROLLMENT_DIR = PROJECT_ROOT / "data" / "enrollment"


# ==========================================
# VALIDATE STUDENT ID
# ==========================================

def validate_student_id(student_id):

    student_id = student_id.strip()

    if not student_id:
        return False

    if not all(
        character.isalnum() or character in "_-"
        for character in student_id
    ):
        return False

    return True


# ==========================================
# SAVE ONE ENROLLED FACE IMAGE
# ==========================================

def save_face_image(
    student_id,
    uploaded_image,
    detector,
    recognizer
):
    """
    Validate, align and save one uploaded face image.
    """

    student_id = student_id.strip()

    # Validate student ID
    if not validate_student_id(student_id):
        return False, "Invalid student ID."

    # Convert uploaded image into OpenCV format
    image_bytes = np.frombuffer(
        uploaded_image.getvalue(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return False, "Could not read the uploaded image."

    # Detect face
    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return False, "No face detected. Try again."

    if len(faces) > 1:
        return False, "Multiple faces detected. Show only one face."

    # Align detected face
    aligned_face = recognizer.alignCrop(
        frame,
        faces[0]
    )

    # Create student enrollment folder
    save_folder = ENROLLMENT_DIR / student_id

    save_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find next available image number
    existing_images = list(
        save_folder.glob("*.jpg")
    )

    count = len(existing_images) + 1

    while (save_folder / f"{count:03d}.jpg").exists():
        count += 1

    image_path = save_folder / f"{count:03d}.jpg"

    # Save aligned face
    success = cv2.imwrite(
        str(image_path),
        aligned_face
    )

    if not success:
        return False, "Could not save face image."

    return True, f"Face image saved successfully: {image_path.name}"
