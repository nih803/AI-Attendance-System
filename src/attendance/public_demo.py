"""Deterministic fictional records for isolated public demonstrations."""

from datetime import date, datetime, timedelta
from contextlib import closing
import random
import sqlite3


STUDENTS = (
    ("DEMO-001", "Avery Morgan"),
    ("DEMO-002", "Jordan Ellis"),
    ("DEMO-003", "Taylor Bennett"),
    ("DEMO-004", "Riley Carter"),
    ("DEMO-005", "Casey Brooks"),
    ("DEMO-006", "Quinn Parker"),
    ("DEMO-007", "Rowan Hayes"),
    ("DEMO-008", "Skyler Reed"),
)


def create_sample_database(db_path, anchor_date=None):
    """Replace the supplied isolated database with reproducible sample data."""
    anchor_date = anchor_date or date.today()
    rng = random.Random(20261115 + anchor_date.toordinal())
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.executescript("""
            DROP TABLE IF EXISTS attendance;
            DROP TABLE IF EXISTS students;
            CREATE TABLE students (student_id TEXT PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                session TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                UNIQUE (student_id, attendance_date, session)
            );
        """)
        connection.executemany("INSERT INTO students VALUES (?, ?)", STUDENTS)
        days = []
        cursor = anchor_date
        while len(days) < 20:
            if cursor.weekday() < 5:
                days.append(cursor)
            cursor -= timedelta(days=1)
        for day in reversed(days):
            for index, (student_id, _) in enumerate(STUDENTS):
                # A few believable absences, stable across reruns for this date.
                if rng.random() < (0.08 + (index % 3) * 0.035):
                    continue
                minute = 5 + rng.randrange(55)
                recorded_at = datetime.combine(day, datetime.min.time()).replace(
                    hour=8, minute=minute
                ).isoformat(sep=" ", timespec="seconds")
                connection.execute(
                    "INSERT INTO attendance"
                    " (student_id, attendance_date, session, recorded_at)"
                    " VALUES (?, ?, ?, ?)",
                    (student_id, day.isoformat(), "Morning", recorded_at),
                )
    return len(STUDENTS), len(days)
