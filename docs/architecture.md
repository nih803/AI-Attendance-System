# AI-Powered Smart Attendance & Student Verification System

## 1. Project Objective

Build a prototype that identifies registered students using face recognition and records attendance automatically. QR verification will provide a controlled fallback, while liveness checks will help detect attempts to fool the camera with photographs or screen replays.

## 2. System Workflow

1. A student stands in front of the webcam.
2. The system detects a face.
3. Face recognition compares it with enrolled students.
4. Liveness checks help assess whether the camera is seeing a live person.
5. If identity is uncertain, the system requests fallback verification or teacher approval.
6. After successful verification, attendance is recorded.
7. Duplicate attendance for the same session is prevented.
8. The dashboard displays attendance records and reports.

## 3. Main Components

- **Face Recognition:** Matches detected faces with registered students.
- **QR Verification:** Provides a controlled fallback; a QR code alone does not prove identity.
- **Liveness Detection:** Helps detect spoofing attempts.
- **Attendance Management:** Stores student details, timestamps and attendance records in SQLite.
- **Dashboard and Reports:** Uses Flask for the interface and supports CSV and Excel exports.

## 4. Planned Technology Stack

- Python — main programming language
- OpenCV — webcam access, image processing and face detection
- Pretrained face-recognition model — identity matching
- SQLite — local database
- Flask — web dashboard
- Pandas and OpenPyXL — reports and Excel exports
- QR library — QR generation and verification

Additional libraries will be selected after checking Python compatibility and model licensing.

## 5. Privacy and Security

- Use consenting participants for testing.
- Keep face images, embeddings, student details, QR secrets and databases out of public Git repositories.
- Do not record attendance when identity is uncertain.
- Do not treat possession of a QR code as proof of identity.
- Test false matches, missed detections and spoofing attempts.

## 6. Current Progress

- Project structure created.
- Git initialized and working code committed.
- Python virtual environment created.
- OpenCV installed.
- Webcam access tested successfully.
- Haar Cascade face detection tested successfully.

## 7. Next Milestone

Choose a compatible pretrained face-recognition model and build the student enrollment and recognition workflow.