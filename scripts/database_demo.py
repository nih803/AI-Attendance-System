
from src.attendance.database import (
    create_tables,
    add_student,
    list_students,
    mark_attendance,
    list_attendance,
)

create_tables()

add_student("DEMO001", "Demo Student")
add_student("DEMO002", "Test Student")
add_student("ST001", "Test Student")
add_student("ST002", "Friend Test")

list_students()

print("\n--- Testing Attendance ---")
mark_attendance("DEMO001", "Morning")
mark_attendance("DEMO001", "Morning")

list_attendance()
