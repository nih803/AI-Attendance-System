# AttendAI — AI-Powered Smart Attendance System

[![Automated tests](https://github.com/nih803/AI-Attendance-System/actions/workflows/tests.yml/badge.svg)](https://github.com/nih803/AI-Attendance-System/actions/workflows/tests.yml)

AttendAI is a local prototype for student enrollment, face matching and classroom attendance. It combines OpenCV YuNet and SFace models, a Streamlit teacher dashboard and a local SQLite database.

> **Prototype notice:** Face recognition can make mistakes and this app does not provide liveness detection, authentication or role-based access. Do not use it as the sole basis for consequential attendance decisions. Get informed consent, provide a manual correction path, and evaluate accuracy and bias with representative, consented data before any real deployment.

## Features

- Register students and enroll face images through the dashboard.
- Reject invalid images, multiple faces and unsafe student IDs.
- Match a single face against locally enrolled references; uncertain matches are not recorded.
- Record one attendance entry per student, date and session in SQLite.
- Browse and filter history, review summaries, and download CSV reports.

## Technology

| Area | Technology |
| --- | --- |
| Dashboard | Streamlit |
| Detection and recognition | OpenCV YuNet and SFace |
| Image arrays | NumPy |
| Reports | Pandas |
| Local storage | SQLite |
| Automated checks | Python `unittest` |

## Setup

### Requirements

- A Python version supported by the pinned dependencies (the current project environment uses Python 3.14)
- A webcam for enrollment and attendance workflows
- YuNet and SFace ONNX model files

### Install

```powershell
git clone https://github.com/nih803/AI-Attendance-System.git
cd AI-Attendance-System
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the OpenCV Zoo models and place these files in `models/`:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

Review the upstream model licenses before redistribution. Model files are excluded from Git; the app reports a readable setup message when they are missing.

### Run

```powershell
streamlit run app.py
```

The app creates its SQLite database when student or attendance tables are first needed. The database is stored at the project root as `attendance.db`.

## Use

1. Open **Students**, register a student, then select them in Face Enrollment.
2. Capture a clear face image and save it. Enrollment photos are stored under `data/enrollment/<student-id>/`.
3. Open **Take Attendance**, capture one face, and select **Recognize & Mark Attendance**.
4. Review entries in **Attendance History** or export reports from **Reports**.

The current session label is `Morning` in `src/attendance/recognition.py`. Change it in code before using a different class session. Recognition thresholds are prototype values and need evaluation for the target camera and population.

## Tests and final checks

Run the offline regression suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The tests use temporary databases and synthetic inputs; they do not need a webcam, enrolled photos or model files. Before a classroom demo, also check:

- Both model files load and the camera is available.
- A consenting enrolled participant is recognized in expected lighting.
- No face, multiple faces, an unknown face and an ambiguous match do not create attendance.
- Repeating the same student/date/session does not create a duplicate.
- History filters, report date ranges and CSV downloads show the expected records.
- `git status --short` contains no student images, local database, model weights, secrets or virtual environment.

## Data handling and repository hygiene

- Student face images and `attendance.db` stay local and are ignored by Git.
- Do not commit photos, biometric templates, student details, exported reports or credentials.
- Limit access to the project folder and delete biometric data when it is no longer needed, in line with your institution's policy.
- The prototype has no sign-in, authorization, encryption-at-rest policy, liveness detection or audit trail. A local-only deployment still requires device and account protections.

See [docs/architecture.md](docs/architecture.md) for data flow, storage boundaries and limitations.
