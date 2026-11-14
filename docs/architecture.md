# AttendAI architecture

## Runtime flow

1. `app.py` starts the Streamlit dashboard and opens the local SQLite database.
2. Student registration is validated in `src/attendance/database.py`.
3. Enrollment in `src/attendance/enrollment.py` checks the student ID, upload size and image dimensions, requires exactly one face, aligns it, then writes a JPEG beneath `data/enrollment/`.
4. `src/attendance/recognition.py` loads the YuNet detector and SFace recognizer from `models/`, builds normalized references from local enrollment JPEGs and compares a captured face against those references.
5. A match over the configured threshold with enough separation from the next candidate may be recorded. SQLite enforces uniqueness for `(student_id, attendance_date, session)`.
6. The dashboard reads history and aggregates records for display and CSV export.

```text
Browser
  └── Streamlit app.py
      ├── enrollment.py ── OpenCV ── data/enrollment/<student-id>/*.jpg
      ├── recognition.py ─ OpenCV YuNet + SFace ─ models/*.onnx
      └── database.py ─── SQLite ── attendance.db
```

## Data boundaries

| Data | Local location | Git policy |
| --- | --- | --- |
| Student face photos | `data/enrollment/` | Ignored |
| Local test images | `data/test_cases/` | Ignored |
| Student and attendance records | `attendance.db` | Ignored |
| Downloaded model weights | `models/*.onnx` | Ignored |
| Test suite | `tests/` | Safe to commit when it contains synthetic fixtures only |

Never commit real student data, biometric images or templates, database exports, credentials, or model weights without checking applicable consent, policy and licensing requirements.

## Recognition and failure behavior

- Images with zero or multiple faces are rejected.
- Unknown or ambiguous candidates do not reach the attendance insert.
- Invalid reference vectors are ignored; missing or incompatible references fail closed.
- Duplicate and unregistered-student attendance outcomes are distinguished.
- UI error messages avoid exposing exception details; diagnostic tracebacks are written to the local app log.
- Similarity and margin thresholds in `recognition.py` are experimental and are not a measured accuracy guarantee.

## Current limits

- This is a local development prototype, without authentication, role-based permissions, transport controls, an audit log or a data-retention workflow.
- There is no robust liveness or presentation-attack detection. A photograph or screen replay may be accepted.
- It uses one configured session label and does not model a timetable. Its attendance percentage is based on recorded date/session data, not all scheduled classes.
- Face recognition can have demographic and environmental error differences. A teacher must review uncertain or disputed results and offer a manual alternative.
- Secure the host account and disk, restrict access to biometric data, and obtain informed consent before enrollment.
