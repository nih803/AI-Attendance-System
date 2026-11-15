"""Offline regression tests for Day 5 database and recognition safeguards."""

from datetime import date
from pathlib import Path
import tempfile
import unittest
import os
import sqlite3
import hashlib
from contextlib import closing
from unittest.mock import patch

import numpy as np

from src.attendance import database, enrollment, recognition, public_demo


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_db_path = database.DB_PATH
        database.DB_PATH = Path(self.temp_dir.name) / "test.db"
        database.create_tables()

    def tearDown(self):
        database.DB_PATH = self.previous_db_path
        self.temp_dir.cleanup()

    def test_student_input_is_trimmed_and_duplicates_are_reported(self):
        self.assertEqual(database.add_student("  S-1 ", "  Ada Lovelace "), "success")
        self.assertEqual(database.add_student("S-1", "Other Name"), "duplicate")
        self.assertEqual(database.list_students(), [("S-1", "Ada Lovelace")])

    def test_empty_or_non_text_student_fields_are_rejected(self):
        self.assertEqual(database.add_student(" ", "Ada"), "invalid")
        self.assertEqual(database.add_student(None, "Ada"), "invalid")

    def test_attendance_duplicate_and_unknown_student_are_distinguished(self):
        self.assertEqual(database.add_student("S-1", "Ada"), "success")
        day = date.today().isoformat()
        self.assertEqual(database.mark_attendance("S-1", "Morning", day), "success")
        self.assertEqual(database.mark_attendance("S-1", "Morning", day), "duplicate")
        self.assertEqual(database.mark_attendance("missing", "Morning", day), "invalid")
        self.assertEqual(len(database.list_attendance()), 1)


class EnrollmentTests(unittest.TestCase):
    def test_student_id_cannot_traverse_enrollment_directory(self):
        self.assertTrue(enrollment.validate_student_id("ST_03-1"))
        self.assertFalse(enrollment.validate_student_id("../student"))
        self.assertFalse(enrollment.validate_student_id(" "))
        self.assertFalse(enrollment.validate_student_id(None))

    def test_invalid_image_is_rejected_before_detector_use(self):
        class UploadedImage:
            def getvalue(self):
                return b"not an image"

        success, message = enrollment.save_face_image(
            "S-1", UploadedImage(), detector=None, recognizer=None
        )
        self.assertFalse(success)
        self.assertIn("Could not read", message)


class RecognitionTests(unittest.TestCase):
    def test_invalid_and_ambiguous_input_never_records_attendance(self):
        frame = np.ones((2, 2, 3), dtype=np.uint8)
        with patch.object(recognition, "mark_attendance") as mark:
            result = recognition.recognize_face(frame, None, None, {})
            self.assertEqual(result["status"], "error")
            processed = recognition.process_recognition_result(result, set())
            self.assertEqual(processed["status"], "error")
            mark.assert_not_called()

    def test_database_failure_does_not_mark_student_processed(self):
        with patch.object(recognition, "mark_attendance", return_value="invalid"):
            marked = set()
            result = recognition.process_recognition_result(
                {"status": "recognized", "student_id": "S-1"}, marked
            )
        self.assertEqual(result["status"], "error")
        self.assertNotIn("S-1", marked)

    def test_successful_attendance_is_marked_once(self):
        with patch.object(recognition, "mark_attendance", return_value="success") as mark:
            marked = set()
            first = recognition.process_recognition_result(
                {"status": "recognized", "student_id": "S-1"}, marked
            )
            second = recognition.process_recognition_result(
                {"status": "recognized", "student_id": "S-1"}, marked
            )
        self.assertEqual(first["status"], "attendance_processed")
        self.assertEqual(second["status"], "already_processed")
        mark.assert_called_once_with("S-1", recognition.SESSION)

    def test_unknown_face_is_never_recognized(self):
        class Detector:
            def setInputSize(self, size):
                pass

            def detect(self, frame):
                return 1, np.ones((1, 15), dtype=np.float32)

        class Recognizer:
            def alignCrop(self, frame, face):
                return frame

            def feature(self, aligned):
                return np.array([[1.0, 0.0]], dtype=np.float32)

        result = recognition.recognize_face(
            np.ones((10, 10, 3), dtype=np.uint8),
            Detector(), Recognizer(), {"S-1": np.array([0.0, 1.0])},
        )
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["student_id"])

    def test_close_candidate_scores_remain_ambiguous_and_unrecorded(self):
        class Detector:
            def setInputSize(self, size):
                pass

            def detect(self, frame):
                return 1, np.ones((1, 15), dtype=np.float32)

        class Recognizer:
            def alignCrop(self, frame, face):
                return frame

            def feature(self, aligned):
                return np.array([1.0, 0.0], dtype=np.float32)

        references = {
            "TEST-001": np.array([0.8, 0.6], dtype=np.float32),
            "TEST-002": np.array([0.8, -0.6], dtype=np.float32),
        }
        with patch.object(recognition, "mark_attendance") as mark:
            result = recognition.recognize_face(
                np.ones((120, 120, 3), dtype=np.uint8),
                Detector(), Recognizer(), references,
            )
            processed = recognition.process_recognition_result(result, set())

        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(processed["status"], "ambiguous")
        mark.assert_not_called()


class ReferenceLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_db_path = database.DB_PATH
        self.previous_enrollment_dir = recognition.ENROLLMENT_DIR
        base = Path(self.temp_dir.name)
        database.DB_PATH = base / "students.db"
        recognition.ENROLLMENT_DIR = base / "enrollment"
        recognition.ENROLLMENT_DIR.mkdir()
        database.create_tables()

        class Detector:
            def setInputSize(self, size):
                pass

            def detect(self, image):
                return 1, np.ones((1, 15), dtype=np.float32)

        class Recognizer:
            def __init__(self, feature):
                self.feature_value = feature

            def alignCrop(self, image, face):
                return image

            def feature(self, aligned):
                return self.feature_value

        self.Detector = Detector
        self.Recognizer = Recognizer
        self.image = np.ones((120, 120, 3), dtype=np.uint8)

    def tearDown(self):
        database.DB_PATH = self.previous_db_path
        recognition.ENROLLMENT_DIR = self.previous_enrollment_dir
        self.temp_dir.cleanup()

    def add_image(self, student_id, filename="001.jpg"):
        folder = recognition.ENROLLMENT_DIR / student_id
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / filename
        image_path.write_bytes(b"synthetic test fixture")
        return image_path

    def test_only_registered_students_with_usable_images_are_loaded(self):
        database.add_student("TEST-001", "Sample Student")
        registered_image = self.add_image("TEST-001")
        self.add_image("DELETED-999")
        detector = self.Detector()
        recognizer = self.Recognizer(np.array([1.0, 0.0], dtype=np.float32))

        with patch.object(recognition.cv2, "imread", return_value=self.image) as read_image:
            with self.assertLogs(recognition.logger, level="INFO") as logs:
                references = recognition.load_student_references(detector, recognizer)

        self.assertEqual(set(references), {"TEST-001"})
        self.assertEqual(read_image.call_count, 1)
        diagnostics = recognition.get_reference_load_diagnostics()
        self.assertEqual(diagnostics["loaded_references"], 1)
        self.assertEqual(diagnostics["loaded_images"], 1)
        self.assertEqual(diagnostics["ignored_folders"], 1)
        self.assertEqual(diagnostics["students_without_references"], 0)
        self.assertNotIn("TEST-001", " ".join(logs.output))
        self.assertTrue(registered_image.exists())
        self.assertEqual(recognition.THRESHOLD, 0.40)
        self.assertEqual(recognition.MARGIN_THRESHOLD, 0.05)

    def test_registered_student_with_missing_images_has_no_reference(self):
        database.add_student("TEST-002", "Sample Student")
        empty_folder = recognition.ENROLLMENT_DIR / "TEST-002"
        empty_folder.mkdir()

        references = recognition.load_student_references(
            self.Detector(), self.Recognizer(np.array([1.0, 0.0]))
        )

        self.assertEqual(references, {})
        diagnostics = recognition.get_reference_load_diagnostics()
        self.assertEqual(diagnostics["loaded_references"], 0)
        self.assertEqual(diagnostics["students_without_references"], 1)

    def test_unreadable_enrollment_image_is_skipped(self):
        database.add_student("TEST-003", "Sample Student")
        self.add_image("TEST-003")

        with patch.object(recognition.cv2, "imread", return_value=None):
            references = recognition.load_student_references(
                self.Detector(), self.Recognizer(np.array([1.0, 0.0]))
            )

        self.assertEqual(references, {})
        diagnostics = recognition.get_reference_load_diagnostics()
        self.assertEqual(diagnostics["skipped_images"], 1)
        self.assertEqual(diagnostics["invalid_features"], 0)

    def test_zero_or_nonfinite_feature_is_rejected_as_invalid(self):
        database.add_student("TEST-004", "Sample Student")
        self.add_image("TEST-004")
        self.add_image("TEST-004", "002.jpg")

        class InvalidFeatureRecognizer(self.Recognizer):
            def __init__(self):
                self.features = [
                    np.array([0.0, 0.0]),
                    np.array([np.nan, 1.0]),
                ]

            def feature(self, aligned):
                return self.features.pop(0)

        with patch.object(recognition.cv2, "imread", return_value=self.image):
            references = recognition.load_student_references(
                self.Detector(), InvalidFeatureRecognizer()
            )

        self.assertEqual(references, {})
        diagnostics = recognition.get_reference_load_diagnostics()
        self.assertEqual(diagnostics["invalid_features"], 2)
        self.assertEqual(diagnostics["loaded_images"], 0)


class PublicDemoTests(unittest.TestCase):
    def test_synthetic_database_is_fictional_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.db"
            second = Path(temp_dir) / "second.db"
            anchor = date(2026, 11, 15)
            public_demo.create_sample_database(first, anchor)
            public_demo.create_sample_database(second, anchor)
            with closing(sqlite3.connect(first)) as connection:
                students = connection.execute(
                    "SELECT student_id, name FROM students ORDER BY student_id"
                ).fetchall()
                records = connection.execute(
                    "SELECT student_id, attendance_date, session, recorded_at "
                    "FROM attendance ORDER BY attendance_date, student_id"
                ).fetchall()
            with closing(sqlite3.connect(second)) as connection:
                other_records = connection.execute(
                    "SELECT student_id, attendance_date, session, recorded_at "
                    "FROM attendance ORDER BY attendance_date, student_id"
                ).fetchall()
            self.assertEqual(len(students), 8)
            self.assertTrue(all(student_id.startswith("DEMO-") for student_id, _ in students))
            self.assertGreater(len(records), 100)
            self.assertEqual(records, other_records)

    def test_public_mode_blocks_database_and_face_writes(self):
        with patch.dict(os.environ, {"ATTENDAI_PUBLIC_DEMO": "true"}):
            with self.assertRaises(PermissionError):
                database.add_student("VISITOR", "Visitor")
            with self.assertRaises(PermissionError):
                database.mark_attendance("DEMO-001", "Morning")
            with self.assertRaises(PermissionError):
                enrollment.save_face_image("DEMO-001", object(), None, None)

    def test_public_app_uses_fictional_session_data_and_hides_write_controls(self):
        from streamlit.testing.v1 import AppTest

        real_db = database.PROJECT_ROOT / "attendance.db"
        before = hashlib.sha256(real_db.read_bytes()).hexdigest() if real_db.exists() else None
        original_db_path = database.DB_PATH
        try:
            with patch.dict(os.environ, {"ATTENDAI_PUBLIC_DEMO": "true"}):
                app_path = Path(__file__).resolve().parents[1] / "app.py"
                app = AppTest.from_file(str(app_path), default_timeout=15).run()
                self.assertEqual(len(app.exception), 0)
                demo_db = Path(app.session_state["public_demo_db_dir"]) / "demo.db"
                self.assertNotEqual(demo_db.resolve(), real_db.resolve())
                with closing(sqlite3.connect(demo_db)) as connection:
                    students = connection.execute(
                        "SELECT student_id, name FROM students"
                    ).fetchall()
                    records = connection.execute(
                        "SELECT student_id FROM attendance"
                    ).fetchall()
                    tables = {
                        row[0] for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    }
                self.assertEqual(len(students), 8)
                self.assertTrue(all(row[0].startswith("DEMO-") for row in students))
                self.assertTrue(all(row[1] in dict(public_demo.STUDENTS).values() for row in students))
                self.assertGreater(len(records), 100)
                self.assertTrue(all(row[0].startswith("DEMO-") for row in records))
                self.assertEqual(tables, {"students", "attendance", "sqlite_sequence"})

                app.radio[0].set_value("Students").run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.form), 0)
                self.assertEqual(len(app.get("camera_input")), 0)
                app.radio[0].set_value("Take Attendance").run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.get("camera_input")), 0)
        finally:
            database.DB_PATH = original_db_path

        after = hashlib.sha256(real_db.read_bytes()).hexdigest() if real_db.exists() else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
