"""Offline regression tests for Day 5 database and recognition safeguards."""

from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.attendance import database, enrollment, recognition


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


if __name__ == "__main__":
    unittest.main()
