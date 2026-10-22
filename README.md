# AI-Powered Smart Attendance System

An AI-based attendance management prototype that identifies enrolled students using face recognition and records their attendance in an SQLite database.

## Project Overview

The goal of this project is to simplify classroom attendance by automating student identification and attendance recording. The system uses OpenCV's pretrained face detection and recognition models to recognize enrolled students and prevent duplicate attendance entries for the same student, date, and session.

## Features Implemented

- **Face enrollment:** Capture and save face images for enrolled students.
- **Automatic face recognition:** Compare a live webcam face against multiple enrolled students.
- **Unknown-face handling:** Reject faces that do not meet the configured recognition criteria.
- **SQLite database:** Store student details and attendance records.
- **Duplicate attendance prevention:** Enforce uniqueness for each student, date, and session.
- **Session-based attendance:** Record attendance for a specified session.
- **Git version control:** Track source code while excluding private data and local model files.

## Planned Features

- QR-based fallback verification.
- Liveness detection and stronger anti-spoofing measures.
- Teacher dashboard using Flask, HTML, CSS, and JavaScript.
- Attendance summaries and CSV/Excel exports.
- Monthly attendance reports and low-attendance notifications.
- Automated present/absent lists and attendance percentages.

## Technology Stack

- **Language:** Python
- **Computer vision:** OpenCV
- **Face detection:** YuNet pretrained model
- **Face recognition:** SFace pretrained model
- **Database:** SQLite
- **Version control:** Git

Flask and frontend technologies are planned for the dashboard; they are not yet part of the implemented core prototype.

## Project Structure

```text
AI-Attendance-System/
├── src/
│   └── attendance/
│       └── database.py
├── data/
│   ├── enrollment/
│   └── test_cases/
├── docs/
│   └── architecture.md
├── models/
├── tests/
├── capture_faces.py
├── recognize_student.py
├── attendance_demo.py
├── model_test.py
├── webcam_test.py
├── face_detection_test.py
├── README.md
└── .gitignore
```

The `.venv/` directory, local database, enrollment images, and downloaded model files are kept out of Git tracking through `.gitignore`.

## Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AI-Attendance-System
```

Replace `<your-repository-url>` with your actual Git repository URL.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install OpenCV

```powershell
python -m pip install opencv-python==4.13.0.92
```

NumPy is also required by the recognition code. If it is not already installed:

```powershell
python -m pip install numpy
```

### 4. Add the pretrained models

Place the required model files in the `models/` directory:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

The model files are not included in Git. Obtain them from the official OpenCV Zoo repository and review their applicable licenses before distributing the project.

## Running the Project

Run these commands from the project root with the virtual environment activated.

### Test the models

```powershell
python model_test.py
```

### Capture student enrollment images

```powershell
python capture_faces.py
```

Follow the prompts to enter a student ID and capture face images. Use only images collected with the student's informed consent.

### Run automatic face recognition

```powershell
python recognize_student.py
```

The system loads enrolled student references, compares detected webcam faces, and attempts to record attendance for accepted matches.

Press the appropriate exit key shown by the program to close the webcam window.

### Test the attendance database

```powershell
python attendance_demo.py
```

This script demonstrates student registration and attendance database operations. Some records may already exist because the SQLite database persists between runs.

## Database Design

The SQLite database contains two main tables:

- **students:** Stores student IDs and names.
- **attendance:** Stores attendance IDs, student IDs, dates, sessions, and recording timestamps.

A uniqueness constraint on the student ID, attendance date, and session prevents duplicate entries for the same session.

## Current Limitations

- Recognition accuracy depends on lighting, camera quality, face position, and enrollment-image quality.
- The configured similarity threshold is experimental and requires testing with representative data.
- The current prototype does not implement robust liveness detection, so it must not be considered resistant to photo or video spoofing.
- QR verification and the teacher dashboard have not yet been implemented.
- Attendance data and recognition results should be reviewed before any real-world deployment.
- Biometric information and student records must be stored securely and accessed only by authorized people.

## Future Enhancements

1. Implement liveness detection and stronger anti-spoofing checks.
2. Add QR verification as a controlled fallback.
3. Build a teacher dashboard with Flask.
4. Add attendance search, summaries, and report exports.
5. Improve recognition evaluation, error handling, and automated testing.
6. Strengthen privacy, access control, and secure data handling.

## Development Status

**Current milestone: Day 6 completed; Day 7 polishing and validation in progress.**

The current prototype supports face enrollment, automatic multi-student recognition, and SQLite-based attendance recording.
