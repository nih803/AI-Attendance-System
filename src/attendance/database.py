
import sqlite3
from pathlib import Path

# Store the database in the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "attendance.db"


def get_connection():
    """Open a connection to the SQLite database."""
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_tables():
    """Create the student and attendance tables."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                session TEXT NOT NULL,
                recorded_at TEXT NOT NULL DEFAULT (
                    datetime('now', 'localtime')
                ),
                FOREIGN KEY (student_id)
                    REFERENCES students(student_id),
                UNIQUE (student_id, attendance_date, session)
            )
        """)

    print("Database tables created successfully!")



def add_student(student_id, name):
    """Register a student in the database."""
    import sqlite3

    try:
        with get_connection() as connection:
            connection.execute(
                "INSERT INTO students (student_id, name) VALUES (?, ?)",
                (student_id, name)
            )
        print(f"Student {name} registered successfully!")

    except sqlite3.IntegrityError:
        print(f"Student ID {student_id} already exists.")


def list_students():
    """Display all registered students."""
    with get_connection() as connection:
        students = connection.execute(
            "SELECT student_id, name FROM students ORDER BY student_id"
        ).fetchall()

    if not students:
        print("No students registered yet.")
        return

    print("\nRegistered Students:")
    for student_id, name in students:
        print(f"ID: {student_id} | Name: {name}")


from datetime import date


def mark_attendance(student_id, session):
    """Record attendance for a student for today's session."""
    attendance_date = date.today().isoformat()

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO attendance
                    (student_id, attendance_date, session)
                VALUES (?, ?, ?)
                """,
                (student_id, attendance_date, session)
            )

        print(f"Attendance marked for {student_id}.")

    except sqlite3.IntegrityError:
        print(
            f"Attendance not recorded for {student_id}. "
            "Check whether the student exists or attendance "
            "has already been marked for this session."
        )


def list_attendance():
    """Display all attendance records."""
    with get_connection() as connection:
        records = connection.execute(
            """
            SELECT student_id, attendance_date, session, recorded_at
            FROM attendance
            ORDER BY attendance_date, student_id
            """
        ).fetchall()

    print("\nAttendance Records:")
    if not records:
        print("No attendance records yet.")
        return

    for student_id, attendance_date, session, recorded_at in records:
        print(
            f"ID: {student_id} | Date: {attendance_date} | "
            f"Session: {session} | Recorded at: {recorded_at}"
        )


if __name__ == "__main__":
    create_tables()
