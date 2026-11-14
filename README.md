# AttendAI — AI-Powered Smart Attendance System

An AI-powered attendance management system that uses face detection and recognition to identify enrolled students and automatically record classroom attendance.

AttendAI combines computer vision, a Streamlit dashboard and an SQLite database to simplify attendance management.

## Project Overview

Traditional classroom attendance is time-consuming and requires manual effort. AttendAI aims to automate this process by identifying registered students through face recognition and recording their attendance.

The system uses OpenCV's pretrained YuNet face detection and SFace face recognition models. Teachers can manage student enrollment, capture face images, take attendance, view attendance history and generate reports through a web-based dashboard.

## Features

### 1. Student Management
- Register students using unique student IDs.
- Capture and enroll student face images.
- Store enrollment images locally.
- View registered students.

### 2. Face Recognition
- Automatic face detection using YuNet.
- Face recognition using SFace.
- Multi-student recognition using enrolled reference images.
- Unknown-face rejection.
- Ambiguous-match handling.

### 3. Attendance Management
- Automatically record attendance after successful recognition.
- Session-based attendance recording.
- Duplicate attendance prevention.
- SQLite database integration.

### 4. Teacher Dashboard
- Interactive Streamlit dashboard.
- Student registration and face enrollment.
- Camera-based attendance marking.
- Attendance history with search and filters.
- Daily and monthly attendance summaries.
- Student-wise attendance percentages.

### 5. Reports and Exports
- Filter reports using a date range.
- View daily attendance summaries.
- View monthly attendance summaries.
- Generate student-wise attendance reports.
- Export attendance reports as CSV files.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Dashboard | Streamlit |
| Computer Vision | OpenCV |
| Face Detection | YuNet |
| Face Recognition | SFace |
| Image Processing | NumPy |
| Data Analysis | Pandas |
| Database | SQLite |
| Version Control | Git and GitHub |

## Project Architecture

```text
AI-Attendance-System/
│
├── app.py
│
├── src/
│   └── attendance/
│       ├── database.py
│       ├── enrollment.py
│       └── recognition.py
│
├── data/
│   ├── enrollment/
│   └── test_cases/
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── docs/
│   └── architecture.md
│
├── attendance_demo.py
├── capture_faces.py
├── recognize_student.py
├── recognize_student_backup.py
├── face_detection_test.py
├── model_test.py
├── webcam_test.py
├── test_enrollment.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

The enrollment images, local database, virtual environment and downloaded ONNX models are excluded from Git tracking.

## Installation and Setup

### Prerequisites

- Python installed on your system.
- Git.
- A webcam.
- Windows, Linux or macOS.

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AI-Attendance-System
```

Replace the placeholder with your GitHub repository URL.

### 2. Create a virtual environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Download pretrained models

Create a `models/` directory and place these model files inside it:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

Obtain the models from the official OpenCV Zoo repository:

https://github.com/opencv/opencv_zoo

Review the applicable model licenses before redistribution.

### 5. Launch AttendAI

Run the following command from the project root:

```powershell
streamlit run app.py
```

The dashboard will open in your browser.

## How to Use AttendAI

1. Open the Streamlit dashboard.
2. Navigate to the Students page.
3. Register a student using their student ID and name.
4. Capture and save their face image.
5. Navigate to Take Attendance.
6. Capture the student's face using the camera.
7. Click Recognize & Mark Attendance.
8. View the recorded attendance in Attendance History.
9. Open Reports to view summaries and download CSV reports.

## Database Design

AttendAI uses SQLite with two main tables:

**Students**
- Student ID
- Student Name

**Attendance**
- Attendance ID
- Student ID
- Attendance Date
- Session
- Recorded Timestamp

A unique constraint on student ID, attendance date and session prevents duplicate attendance entries.

## Current Limitations

- Recognition accuracy depends on lighting, camera quality, face position and enrollment-image quality.
- Recognition thresholds are experimental and require further evaluation.
- Robust liveness detection and anti-spoofing are not yet implemented.
- QR-based fallback verification is not yet implemented.
- Attendance percentages currently use recorded date-session combinations as their denominator, rather than a separate timetable of all scheduled classes.
- The prototype does not yet include authentication or role-based access control.

This project is currently a local development prototype and should not be used as a production biometric attendance system without further security, privacy and accuracy testing.

## Privacy and Security

- Student enrollment images are stored locally.
- Local databases and downloaded models are excluded from Git tracking.
- Enrollment should only be performed with informed student consent.
- Access to biometric information and attendance records should be restricted.

## Future Enhancements

1. Implement liveness detection and stronger anti-spoofing checks.
2. Add QR-based fallback verification.
3. Implement authentication and role-based access control.
4. Add scheduled class sessions for more accurate attendance percentages.
5. Improve recognition accuracy through systematic evaluation.
6. Add Excel exports and automated low-attendance notifications.
7. Strengthen privacy and secure data handling.

## Development Status

**Phase 2 — Day 4 completed; Day 5 final polishing in progress.**

The current prototype supports student registration, face enrollment, automatic face recognition, SQLite-based attendance recording, a Streamlit teacher dashboard, attendance history, reporting and CSV exports.