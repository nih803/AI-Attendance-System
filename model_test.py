import cv2
from pathlib import Path

# Paths to our pretrained models
yunet_path = Path("models/face_detection_yunet_2023mar.onnx")
sface_path = Path("models/face_recognition_sface_2021dec.onnx")

# Check that both files exist
if not yunet_path.is_file() or not sface_path.is_file():
    raise FileNotFoundError("One or both model files are missing.")

# Load the YuNet face detector
face_detector = cv2.FaceDetectorYN_create(
    str(yunet_path),
    "",
    (320, 320)
)

# Load the SFace face recognizer
face_recognizer = cv2.FaceRecognizerSF_create(
    str(sface_path),
    ""
)

print("YuNet face detector loaded successfully!")
print("SFace face recognizer loaded successfully!")
print("Both models are ready for testing.")