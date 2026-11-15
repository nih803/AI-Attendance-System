# AttendAI — AI-Powered Smart Attendance System

**Recognize. Record. Report.**

An AI-powered smart attendance management system that automates classroom attendance using facial detection and recognition. AttendAI combines computer vision, student enrollment, automated attendance recording, database management, and reporting into a unified teacher-oriented application.

The system uses OpenCV's YuNet face detection model and SFace face recognition model to identify enrolled students from camera input. It provides an interactive Streamlit dashboard for managing students, taking attendance, reviewing attendance history, and generating reports.

**Prototype disclaimer:** AttendAI is a portfolio and demonstration prototype. It demonstrates the integration of computer vision, database management, and application development. It has not been validated for high-stakes institutional attendance or production biometric identification. Recognition performance depends on image quality, lighting, camera conditions, and model thresholds.

## What the system does

- Registers students and maintains student records.
- Enrolls student face images after validating uploaded images.
- Detects faces using the OpenCV YuNet model.
- Recognizes enrolled students using the SFace face recognition model.
- Automatically records attendance for recognized students.
- Prevents duplicate attendance entries for the same student, date, and session.
- Maintains attendance history with student search and date/session filtering.
- Generates daily, monthly, and student-wise attendance summaries.
- Exports attendance reports in CSV format.
- Provides a centralized teacher-oriented dashboard.
- Uses SQLite for persistent student and attendance records.
- Includes automated regression tests and GitHub Actions verification.

## System workflow

1. **Student registration:** The teacher registers a student using their student ID and name.
2. **Face enrollment:** A face image is uploaded or captured. The system validates the image and detects exactly one face before saving the aligned face image.
3. **Face detection:** During attendance, YuNet detects faces from camera input.
4. **Face recognition:** SFace extracts facial features and compares them against enrolled student references.
5. **Attendance recording:** A recognized student is recorded in the SQLite database.
6. **Duplicate prevention:** Database constraints prevent duplicate attendance for the same student, date, and session.
7. **Attendance reporting:** Teachers can review records, filter attendance, view summaries, and export CSV reports.

## Technology stack

| Layer | Technologies |
|---|---|
| Application interface | Python, Streamlit |
| Programming language | Python |
| Computer vision | OpenCV |
| Face detection | YuNet |
| Face recognition | SFace |
| Image processing | OpenCV, NumPy, Pillow |
| Database | SQLite |
| Data processing | Pandas, NumPy |
| Model format | ONNX |
| Testing | Python unittest, regression tests |
| Version control | Git, GitHub |
| Continuous integration | GitHub Actions |

## Project structure

```text
AI-Attendance-System/
|
|-- app.py                         Main Streamlit application
|
|-- src/
|   `-- attendance/
|       |-- database.py             SQLite database operations
|       |-- enrollment.py           Student face enrollment
|       `-- recognition.py          Face detection and recognition
|
|-- scripts/
|   |-- database_demo.py            Database demonstration
|   |-- enrollment_camera_demo.py   Enrollment camera testing
|   |-- enrollment_webcam.py        Webcam enrollment utility
|   |-- face_detection_demo.py     Face detection demonstration
|   |-- model_check.py              Model and API verification
|   |-- recognition_demo.py         Recognition demonstration
|   `-- webcam_check.py             Webcam verification
|
|-- tests/
|   `-- test_day5.py                Day 5 regression tests
|
|-- models/                         Downloaded ONNX model files
|
|-- data/
|   `-- enrollment/                 Locally stored student face images
|
|-- docs/
|   `-- architecture.md             System architecture documentation
|
|-- .github/
|   `-- workflows/                  GitHub Actions CI workflow
|
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Prerequisites

- Python 3.10 or newer.
- Git.
- A webcam for live face recognition.
- Windows, macOS, or Linux.
- The required ONNX model files.

## Run locally on Windows

### 1. Clone the repository

Open PowerShell:

```powershell
git clone https://github.com/nih803/AI-Attendance-System.git

cd AI-Attendance-System
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Configure the face recognition models

Download the compatible YuNet and SFace ONNX model files from the official OpenCV Zoo model repository.

Place the files in the `models/` directory using these exact filenames:

```text
models/
|-- face_detection_yunet_2023mar.onnx
`-- face_recognition_sface_2021dec.onnx
```

The model files are excluded from GitHub to avoid committing large binary files. They must be supplied locally before running face enrollment and recognition.

### 5. Run the application

From the project root:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

The application will open in your browser.

## Application modules

### 1. Student management and enrollment

The enrollment module handles student face-image processing.

Key operations:

- Student ID validation.
- Uploaded image decoding.
- Image-size and dimension validation.
- Face detection.
- Rejection of images containing zero or multiple faces.
- Face alignment using facial landmarks.
- Storage of aligned face images in student-specific folders.

### 2. Face detection and recognition

AttendAI uses two pretrained computer vision models.

**YuNet — Face Detection**

- Detects faces in camera frames.
- Provides facial landmarks.
- Identifies face locations for subsequent processing.

**SFace — Face Recognition**

- Generates facial feature representations.
- Compares detected faces against enrolled references.
- Uses a similarity threshold to determine whether a match is acceptable.

The recognition threshold is a configurable parameter and requires calibration against representative real-world images.

### 3. Attendance database

SQLite provides local persistent storage.

The database maintains student information and attendance records.

A unique database constraint on:

```sql
(student_id, attendance_date, session)
```

prevents duplicate attendance entries for the same student, date, and session.

### 4. Attendance history and reports

The reporting module provides:

- Student-wise attendance history.
- Date-based filtering.
- Session-based filtering.
- Daily attendance summaries.
- Monthly attendance summaries.
- Student-wise attendance reports.
- CSV report exports.

## Testing and verification

AttendAI includes automated regression tests for important application behavior.

Run the test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

Run Python syntax checks:

```powershell
python -m compileall -q app.py src scripts tests
```

Check Git whitespace errors:

```powershell
git diff --check
```

The project also includes a GitHub Actions workflow for automated verification when changes are pushed.

**Latest development verification:** The Day 5 development report recorded nine passing offline regression tests and a successful GitHub Actions run. Live webcam and model-based verification should be performed separately.

## Security and privacy

Student face images and attendance records are sensitive information. The project includes basic privacy and input-validation measures:

- Face images are stored locally.
- Enrollment images are validated before storage.
- Uploaded image size is restricted.
- Image dimensions are checked.
- Invalid and multiple-face enrollment images are rejected.
- SQL queries use parameterized values.
- Database constraints prevent duplicate attendance.
- Local databases, enrollment images, ONNX model files, and local configuration files are excluded from Git using `.gitignore`.
- Regression tests help verify important application behavior.

**Important production limitations:**

- The prototype does not currently provide a complete authentication and role-based access-control system.
- Stored biometric data does not have a complete encryption-at-rest solution.
- Liveness detection and advanced anti-spoofing are not implemented.
- Recognition thresholds require further evaluation under different lighting, camera, and environmental conditions.
- The local SQLite database is intended for prototype usage, not concurrent large-scale institutional deployment.

## Future enhancements

- Implement liveness detection and stronger anti-spoofing mechanisms.
- Add teacher authentication and role-based access control.
- Introduce stronger biometric data protection and encryption.
- Develop a browser-based camera capture workflow for cloud deployment.
- Add advanced attendance analytics and visualizations.
- Support automated low-attendance notifications.
- Evaluate recognition accuracy using a representative test dataset.
- Explore cloud database integration and scalable deployment.
- Improve automated testing and monitoring.

## Repository

**GitHub:** https://github.com/nih803/AI-Attendance-System

**Project:** AttendAI — AI-Powered Smart Attendance System

**Developed using:** Python, OpenCV, YuNet, SFace, Streamlit, and SQLite.

## Public live demo

**Live demo:** _Not deployed yet — URL will be added after a successful deployment._

The hosted public version runs in read-only demo mode when `ATTENDAI_PUBLIC_DEMO=true` is set as an environment variable or Streamlit secret. It automatically creates a reproducible sample roster and attendance history in a temporary SQLite database allocated to that browser session. The sample data is regenerated as needed and is not stored in the repository or in `attendance.db`.

**Synthetic data disclaimer:** Every student name, ID, date, and attendance record in public demo mode is fictional. No actual student records, face images, or biometric information are included. The interface labels the records as fictional demonstration data and displays a read-only notice.

Visitors can explore the dashboard, student roster, searchable and filterable attendance history, reports, charts, and CSV downloads. Student registration, face enrollment, camera input, recognition, and attendance recording are unavailable in the hosted read-only version. The public app does not load ONNX models. In local mode, YuNet detects faces and SFace compares features against locally enrolled references; only a sufficiently confident, non-ambiguous result is recorded. Recognition results are never simulated in the public demo.

## Deploy on Streamlit Community Cloud

1. Push the `cloud-demo` branch to GitHub after reviewing it.
2. In Streamlit Community Cloud, create an app from `nih803/AI-Attendance-System`, select the `cloud-demo` branch, and set `app.py` as the entry point.
3. In the app's **Settings → Secrets**, add:

   ```toml
   ATTENDAI_PUBLIC_DEMO = "true"
   ```

4. Deploy. The app uses the root `requirements.txt`; do not add ONNX model files for public demo mode.
5. Confirm the read-only notice, fictional sample records, reports, and CSV downloads are visible. Add the resulting app URL above only after deployment succeeds.

The temporary demo database is isolated per visitor session and may be discarded when the session or cloud instance ends. This is expected: it is recreated automatically from the sample generator.

## Local full application setup

For the full local workflow, leave `ATTENDAI_PUBLIC_DEMO` unset or set it to `false`, install the dependencies and compatible ONNX files as described in [Run locally on Windows](#run-locally-on-windows), then run `streamlit run app.py`. Local mode continues to use the project `attendance.db` and the full student registration, face enrollment, and real recognition attendance workflow. Keep real data and face images private and out of Git.
