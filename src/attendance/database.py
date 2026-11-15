"""SQLite helpers for AttendAI student and attendance records."""

from datetime import date
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import os


def public_demo_enabled():
    """Read the public-demo switch from environment or Streamlit secrets."""
    if os.getenv("ATTENDAI_PUBLIC_DEMO", "").strip().lower() == "true":
        return True
    try:
        import streamlit as st
        return str(st.secrets.get("ATTENDAI_PUBLIC_DEMO", "")).strip().lower() == "true"
    except Exception:
        return False


def _require_local_write_access():
    if public_demo_enabled():
        raise PermissionError("Write operations are disabled in public demo mode.")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "attendance.db"


@contextmanager
def get_connection():
    """Yield a connection with foreign keys enabled and always close it."""
    connection = sqlite3.connect(DB_PATH, timeout=10)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            yield connection
    finally:
        connection.close()


def create_tables():
    """Create the student and attendance tables if they do not exist."""
    _require_local_write_access()
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL CHECK (length(trim(name)) > 0)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                session TEXT NOT NULL CHECK (length(trim(session)) > 0),
                recorded_at TEXT NOT NULL DEFAULT (
                    datetime('now', 'localtime')
                ),
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                UNIQUE (student_id, attendance_date, session)
            )
            """
        )


def add_student(student_id, name):
    """Register a student; return ``success``, ``duplicate`` or ``invalid``."""
    _require_local_write_access()
    if not isinstance(student_id, str) or not isinstance(name, str):
        return "invalid"

    student_id, name = student_id.strip(), name.strip()
    if not student_id or not name:
        return "invalid"

    try:
        with get_connection() as connection:
            connection.execute(
                "INSERT INTO students (student_id, name) VALUES (?, ?)",
                (student_id, name),
            )
        return "success"
    except sqlite3.IntegrityError:
        return "duplicate"


def list_students():
    """Return students ordered by ID."""
    with get_connection() as connection:
        return connection.execute(
            "SELECT student_id, name FROM students ORDER BY student_id"
        ).fetchall()


def mark_attendance(student_id, session, attendance_date=None):
    """Record attendance; return ``success``, ``duplicate`` or ``invalid``."""
    _require_local_write_access()
    if not isinstance(student_id, str) or not student_id.strip():
        return "invalid"
    if not isinstance(session, str) or not session.strip():
        return "invalid"

    attendance_date = attendance_date or date.today().isoformat()
    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO attendance (student_id, attendance_date, session)
                VALUES (?, ?, ?)
                """,
                (student_id.strip(), attendance_date, session.strip()),
            )
        return "success"
    except sqlite3.IntegrityError as error:
        # A foreign-key failure means the recognized ID is not registered.
        if "FOREIGN KEY" in str(error).upper():
            return "invalid"
        return "duplicate"


def list_attendance():
    """Return attendance rows ordered by date and student."""
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT student_id, attendance_date, session, recorded_at
            FROM attendance
            ORDER BY attendance_date, student_id
            """
        ).fetchall()


if __name__ == "__main__":
    create_tables()
