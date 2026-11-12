import streamlit as st
import sqlite3
from pathlib import Path
from datetime import date, datetime
import cv2
import numpy as np

from src.attendance.recognition import (
    load_models,
    load_student_references,
    recognize_face,
    process_recognition_result
)

from src.attendance.database import add_student
from src.attendance.enrollment import save_face_image

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AttendAI | Smart Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DATABASE CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "attendance.db"
# ==========================================
# LOAD FACE RECOGNITION MODELS
# ==========================================

@st.cache_resource
def initialize_recognition():

    detector, recognizer = load_models()

    student_references = load_student_references(
        detector,
        recognizer
    )

    return detector, recognizer, student_references


def get_dashboard_data():
    """Read dashboard statistics and attendance records."""

    if not DB_PATH.exists():
        return {
            "total_students": 0,
            "present_today": 0,
            "absent_today": 0,
            "attendance_rate": 0,
            "recent_records": []
        }

    with sqlite3.connect(DB_PATH) as connection:
        total_students = connection.execute(
            "SELECT COUNT(*) FROM students"
        ).fetchone()[0]

        today = date.today().isoformat()

        present_today = connection.execute(
            """
            SELECT COUNT(DISTINCT student_id)
            FROM attendance
            WHERE attendance_date = ?
            """,
            (today,)
        ).fetchone()[0]

        absent_today = max(total_students - present_today, 0)

        attendance_rate = (
            round((present_today / total_students) * 100, 1)
            if total_students > 0
            else 0
        )

        recent_records = connection.execute(
            """
            SELECT
                s.student_id,
                s.name,
                a.attendance_date,
                a.session,
                a.recorded_at
            FROM attendance AS a
            JOIN students AS s
                ON a.student_id = s.student_id
            ORDER BY a.attendance_date DESC,
                     a.recorded_at DESC
            LIMIT 10
            """
        ).fetchall()

    return {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_rate": attendance_rate,
        "recent_records": recent_records
    }


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #F5F7FB;
    color: #17243A;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #17243A;
}

section[data-testid="stSidebar"] * {
    color: #F5F7FB;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: transparent;
    border-radius: 8px;
    padding: 9px 12px;
}

/* Headings */
h1, h2, h3 {
    color: #17243A;
}

/* Statistic cards */
.stat-card {
    background-color: #FFFFFF;
    border: 1px solid #E6EAF1;
    border-radius: 14px;
    padding: 22px;
    min-height: 145px;
    box-shadow: 0 3px 12px rgba(23, 36, 58, 0.04);
}

.stat-label {
    color: #66748A;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 12px;
}

.stat-value {
    color: #17243A;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 7px;
}

.stat-note {
    color: #718096;
    font-size: 12px;
}

/* Welcome banner */
.welcome-banner {
    background: linear-gradient(120deg, #17243A, #244B70);
    color: #FFFFFF;
    padding: 27px 30px;
    border-radius: 16px;
    margin-bottom: 25px;
}

.welcome-title {
    font-size: 25px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 8px;
}

.welcome-text {
    color: #DCE7F3;
    font-size: 14px;
}

/* Section heading */
.section-heading {
    font-size: 20px;
    font-weight: 700;
    color: #17243A;
    margin-top: 20px;
    margin-bottom: 12px;
}

/* ==========================================
   ALL BUTTONS — WHITE TEXT
   ========================================== */

/* Regular buttons and download buttons */
.stButton > button,
.stDownloadButton > button,
[data-testid="stDownloadButton"] button {
    color: #FFFFFF !important;
    background-color: #17243A !important;
    border: 1px solid #2D405A !important;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}

/* Force all nested button content to white */
.stButton > button *,
.stDownloadButton > button *,
[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
}

/* Keep button text visible */
.stButton > button p,
.stDownloadButton > button p,
[data-testid="stDownloadButton"] button p {
    color: #FFFFFF !important;
}

/* Button hover effects */
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    background-color: #244B70 !important;
    border-color: #3B6289 !important;
    color: #FFFFFF !important;
}

/* Preserve white text when hovering over nested elements */
.stButton > button:hover *,
.stDownloadButton > button:hover *,
[data-testid="stDownloadButton"] button:hover * {
    color: #FFFFFF !important;
}

/* Keyboard focus */
.stButton > button:focus,
.stDownloadButton > button:focus,
[data-testid="stDownloadButton"] button:focus {
    color: #FFFFFF !important;
    border-color: #5B9BD5 !important;
}

/* Take Photo button: change text color only; preserve its existing background */
[data-testid="stCameraInput"] button,
[data-testid="stCameraInput"] button p,
[data-testid="stCameraInput"] button span {
    color: #FFFFFF !important;
}

/* Make Streamlit alert messages readable */
[data-testid="stAlert"],
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #17243A !important;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #E6EAF1;
    border-radius: 12px;
    overflow: hidden;
}

/* Hide Streamlit footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("# 🎓 AttendAI")
    st.caption("SMART ATTENDANCE SYSTEM")
    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Students",
            "Take Attendance",
            "Attendance History",
            "Reports",
            "Settings"
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.caption("AI-Powered Smart Attendance")
    st.caption("Local prototype • Development mode")


# Keep the selected student available across Streamlit reruns.
if "enrollment_student_id" not in st.session_state:
    st.session_state.enrollment_student_id = ""

if "last_saved_face_hash" not in st.session_state:
    st.session_state.last_saved_face_hash = ""


# ==========================================
# LOAD DATA
# ==========================================
data = get_dashboard_data()

total_students = data["total_students"]
present_today = data["present_today"]
absent_today = data["absent_today"]
attendance_rate = data["attendance_rate"]
recent_records = data["recent_records"]


# ==========================================
# DASHBOARD PAGE
# ==========================================
if page == "Dashboard":

    current_date = datetime.now().strftime("%A, %d %B %Y")

    st.markdown(f"""
    <div class="welcome-banner">
        <div class="welcome-title">
            Welcome to AttendAI 👋
        </div>
        <div class="welcome-text">
            Your classroom attendance overview · {current_date}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.title("Dashboard")
    st.caption("Monitor student attendance and recent activity.")

    if st.button("↻ Refresh Dashboard"):
        st.rerun()

    st.markdown("### Today's Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">👥 Total Students</div>
            <div class="stat-value">{total_students}</div>
            <div class="stat-note">Registered students</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">✅ Present Today</div>
            <div class="stat-value">{present_today}</div>
            <div class="stat-note">Students with a record today</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">🕒 No Record Today</div>
            <div class="stat-value">{absent_today}</div>
            <div class="stat-note">Students without a record today</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">📊 Attendance Rate</div>
            <div class="stat-value">{attendance_rate}%</div>
            <div class="stat-note">Based on today's records</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([2, 1])

    with left:
        st.markdown(
            '<div class="section-heading">Recent Attendance</div>',
            unsafe_allow_html=True
        )

        if recent_records:
            table_data = [
                {
                    "Student ID": row[0],
                    "Student Name": row[1],
                    "Date": row[2],
                    "Session": row[3],
                    "Recorded At": row[4]
                }
                for row in recent_records
            ]

            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No attendance records available yet.")

    with right:
        st.markdown(
            '<div class="section-heading">Quick Overview</div>',
            unsafe_allow_html=True
        )

        st.markdown("**Today's attendance coverage**")
        st.progress(attendance_rate / 100)

        st.write(
            f"**{attendance_rate}%** of registered students "
            "have an attendance record today."
        )

        st.markdown("---")
        st.markdown("**Quick navigation**")

        if st.button("👥 View Students", use_container_width=True):
            st.session_state["requested_page"] = "Students"
            st.rerun()

        if st.button(
            "📋 Attendance History",
            use_container_width=True
        ):
            st.session_state["requested_page"] = "Attendance History"
            st.rerun()


# ==========================================
# STUDENTS PAGE
# ==========================================
elif page == "Students":

    st.title("👥 Student Management")
    st.caption("Register students and enroll their face images.")

    st.subheader("Register a New Student")

    with st.form("student_registration_form"):
        student_id_input = st.text_input(
            "Student ID",
            placeholder="Example: ST003"
        )
        student_name_input = st.text_input(
            "Student Name",
            placeholder="Enter full name"
        )
        register_clicked = st.form_submit_button(
            "Register Student",
            use_container_width=True
        )

    if register_clicked:
        clean_id = student_id_input.strip()
        clean_name = student_name_input.strip()

        if not clean_id or not clean_name:
            st.error("Please enter both Student ID and Student Name.")
        else:
            registration_status = add_student(clean_id, clean_name)

            if registration_status == "success":
                st.session_state.enrollment_student_id = clean_id
                st.session_state.last_saved_face_hash = ""
                st.success(
                    f"{clean_name} ({clean_id}) registered successfully. "
                    "You can now enroll face images below."
                )
                st.rerun()
            elif registration_status == "duplicate":
                st.warning(
                    "This Student ID already exists. "
                    "Choose a different ID or select the existing student below."
                )
            else:
                st.error("Invalid student details.")

    st.divider()
    st.subheader("Face Enrollment")

    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as connection:
            students = connection.execute(
                """
                SELECT student_id, name
                FROM students
                ORDER BY student_id
                """
            ).fetchall()
    else:
        students = []

    if students:
        student_options = {
            f"{student_id} — {name}": student_id
            for student_id, name in students
        }
        option_labels = list(student_options.keys())

        current_id = st.session_state.enrollment_student_id
        selected_index = 0
        for index, label in enumerate(option_labels):
            if student_options[label] == current_id:
                selected_index = index
                break

        selected_label = st.selectbox(
            "Select student to enroll",
            option_labels,
            index=selected_index,
            key="enrollment_student_selector"
        )
        selected_student_id = student_options[selected_label]
        st.session_state.enrollment_student_id = selected_student_id

        enrollment_folder = (
            PROJECT_ROOT / "data" / "enrollment" / selected_student_id
        )
        existing_face_count = (
            len(list(enrollment_folder.glob("*.jpg")))
            if enrollment_folder.exists()
            else 0
        )

        st.info(
            f"Selected student: **{selected_student_id}** · "
            f"Saved face images: **{existing_face_count}**"
        )
        st.write(
            "Capture one clear face photo at a time. Change your head "
            "angle slightly between photos. Only one person should be visible."
        )

        captured_face = st.camera_input(
            "Capture student face",
            key=f"enrollment_camera_{selected_student_id}"
        )

        if captured_face is not None:
            import hashlib

            image_hash = hashlib.sha256(
                captured_face.getvalue()
            ).hexdigest()

            if image_hash == st.session_state.last_saved_face_hash:
                st.caption(
                    "This photo has already been saved. "
                    "Use Retake photo to capture another angle."
                )
            elif st.button(
                "Save Face Image",
                key=f"save_face_{selected_student_id}",
                type="primary"
            ):
                try:
                    detector, recognizer, _ = initialize_recognition()
                    success, message = save_face_image(
                        selected_student_id,
                        captured_face,
                        detector,
                        recognizer
                    )

                    if success:
                        st.session_state.last_saved_face_hash = image_hash
                        initialize_recognition.clear()
                        st.success(message)
                        st.rerun()
                    else:
                        st.warning(message)

                except Exception as error:
                    st.error(f"Face enrollment failed: {error}")

        st.caption(
            "Tip: Capture several photos with small changes in head angle "
            "and lighting to improve recognition reliability."
        )
    else:
        st.info(
            "Register a student first. The student will then appear "
            "in the face-enrollment selector."
        )

    st.divider()
    st.subheader("Registered Students")

    if students:
        st.dataframe(
            [
                {"Student ID": student_id, "Student Name": name}
                for student_id, name in students
            ],
            use_container_width=True,
            hide_index=True
        )
        st.metric("Registered Students", len(students))
    else:
        st.info("No students registered yet.")


# ==========================================
# TAKE ATTENDANCE PAGE
# ==========================================
elif page == "Take Attendance":

    st.title("📸 Take Attendance")

    st.write(
        "Capture a student's face to recognize them and record attendance."
    )

    st.info(
        "Position one enrolled student's face clearly in the camera. "
        "After taking the photo, click Recognize & Mark Attendance."
    )

    try:
        detector, recognizer, student_references = initialize_recognition()

        if not student_references:
            st.warning(
                "No enrolled student face images were found. "
                "Please enroll students before taking attendance."
            )
        else:
            st.caption(
                f"Recognition system ready · "
                f"{len(student_references)} enrolled student(s) loaded"
            )

            captured_image = st.camera_input("Take Photo")

            if captured_image is not None:
                if st.button(
                    "Recognize & Mark Attendance",
                    key="recognize_mark_attendance",
                    type="primary"
                ):
                    image_bytes = np.frombuffer(
                        captured_image.getvalue(),
                        dtype=np.uint8
                    )
                    frame = cv2.imdecode(
                        image_bytes,
                        cv2.IMREAD_COLOR
                    )

                    if frame is None:
                        st.error("Could not read the captured photo. Please try again.")
                    else:
                        with st.spinner("Recognizing face and recording attendance..."):
                            result = recognize_face(
                                frame,
                                detector,
                                recognizer,
                                student_references
                            )

                            attendance_result = process_recognition_result(
                                result,
                                st.session_state.setdefault("attendance_marked", set())
                            )

                        status = attendance_result.get("status")
                        message = attendance_result.get("message", "")

                        if status == "attendance_processed":
                            st.success(message)
                        elif status in ("already_processed", "duplicate_or_invalid"):
                            st.warning(message)
                        elif status == "no_face":
                            st.warning("No face detected. Please capture another photo with your face clearly visible.")
                        elif status == "multiple_faces":
                            st.warning("Multiple faces detected. Please make sure only one person is visible.")
                        elif status == "unknown":
                            st.error(result.get("message", "Unknown face."))
                        elif status == "ambiguous":
                            st.warning(result.get("message", "Ambiguous match. Attendance was not marked."))
                        elif status == "recognized":
                            st.success(result.get("message", "Face recognized."))
                        else:
                            st.error(message or result.get("message", "Recognition failed."))

                        similarity = result.get("similarity")
                        if similarity is not None:
                            st.caption(f"Recognition similarity: {similarity:.2f}")

    except Exception as error:
        st.error(f"Could not start the recognition system: {error}")


# ==========================================
# ATTENDANCE HISTORY PAGE
# ==========================================
elif page == "Attendance History":

    st.title("📋 Attendance History")
    st.caption("Search, filter and browse attendance records.")

    if DB_PATH.exists():

        with sqlite3.connect(DB_PATH) as connection:
            records = connection.execute(
                """
                SELECT
                    s.student_id,
                    s.name,
                    a.attendance_date,
                    a.session,
                    a.recorded_at
                FROM attendance AS a
                JOIN students AS s
                    ON a.student_id = s.student_id
                ORDER BY a.attendance_date DESC,
                         a.recorded_at DESC
                """
            ).fetchall()

        if records:

            # Convert database records into dictionaries
            attendance_data = [
                {
                    "Student ID": row[0],
                    "Student Name": row[1],
                    "Date": row[2],
                    "Session": row[3],
                    "Recorded At": row[4]
                }
                for row in records
            ]

            st.subheader("Search Attendance")

            # Search by student ID or name
            search_query = st.text_input(
                "Search student",
                placeholder="Enter student ID or student name..."
            )

            col1, col2 = st.columns(2)

            # Get available dates and sessions
            available_dates = sorted(
                {row["Date"] for row in attendance_data},
                reverse=True
            )

            available_sessions = sorted(
                {row["Session"] for row in attendance_data}
            )

            with col1:
                selected_date = st.selectbox(
                    "Filter by Date",
                    ["All dates"] + available_dates
                )

            with col2:
                selected_session = st.selectbox(
                    "Filter by Session",
                    ["All sessions"] + available_sessions
                )

            # Apply filters
            filtered_records = attendance_data

            if search_query.strip():

                query = search_query.strip().lower()

                filtered_records = [
                    row for row in filtered_records
                    if query in row["Student ID"].lower()
                    or query in row["Student Name"].lower()
                ]

            if selected_date != "All dates":

                filtered_records = [
                    row for row in filtered_records
                    if row["Date"] == selected_date
                ]

            if selected_session != "All sessions":

                filtered_records = [
                    row for row in filtered_records
                    if row["Session"] == selected_session
                ]

            st.divider()

            # Display filtered results
            st.subheader("Attendance Records")

            st.metric(
                "Matching Records",
                len(filtered_records)
            )

            if filtered_records:

                st.dataframe(
                    filtered_records,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.warning(
                    "No attendance records match your filters."
                )

        else:

            st.info("No attendance records found.")

    else:

        st.warning(
            "The attendance database has not been created."
        )


# ==========================================
# REPORTS PAGE
# ==========================================
elif page == "Reports":

    import pandas as pd

    st.title("📊 Attendance Reports")
    st.caption("Analyze daily, monthly and student-wise attendance.")

    if DB_PATH.exists():

        st.subheader("Filter Report by Date")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date.today().replace(day=1)
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=date.today()
            )

        if start_date > end_date:
            st.error("Start date cannot be after end date.")
            st.stop()

        with sqlite3.connect(DB_PATH) as connection:

            records = connection.execute(
                """
                SELECT
                    s.student_id,
                    s.name,
                    a.attendance_date,
                    a.session,
                    a.recorded_at
                FROM attendance AS a
                JOIN students AS s
                    ON a.student_id = s.student_id
                ORDER BY a.attendance_date DESC
                """
            ).fetchall()

            students = connection.execute(
                """
                SELECT student_id, name
                FROM students
                ORDER BY student_id
                """
            ).fetchall()

        if records:

            df = pd.DataFrame(
                records,
                columns=[
                    "Student ID",
                    "Student Name",
                    "Date",
                    "Session",
                    "Recorded At"
                ]
            )

            df["Date"] = pd.to_datetime(df["Date"])

            # Filter records using selected dates
            df = df[
                (df["Date"].dt.date >= start_date) &
                (df["Date"].dt.date <= end_date)
            ]

            if df.empty:
                st.info(
                    "No attendance records found for the selected date range."
                )
                st.stop()

            # --------------------------------
            # DAILY SUMMARY
            # --------------------------------

            st.subheader("Daily Attendance Summary")

            daily_summary = (
                df.groupby("Date")
                .agg(
                    Present=("Student ID", "nunique"),
                    Sessions=("Session", "nunique")
                )
                .reset_index()
            )

            daily_summary["Date"] = (
                daily_summary["Date"].dt.strftime("%Y-%m-%d")
            )

            st.dataframe(
                daily_summary,
                width="stretch",
                hide_index=True
            )

            # --------------------------------
            # MONTHLY SUMMARY
            # --------------------------------

            st.divider()
            st.subheader("Monthly Attendance Summary")

            df["Month"] = df["Date"].dt.strftime("%Y-%m")

            monthly_summary = (
                df.groupby("Month")
                .agg(
                    Total_Attendance_Records=("Student ID", "count"),
                    Unique_Students=("Student ID", "nunique")
                )
                .reset_index()
            )

            st.dataframe(
                monthly_summary,
                width="stretch",
                hide_index=True
            )

            # --------------------------------
            # STUDENT-WISE ATTENDANCE
            # --------------------------------

            st.divider()
            st.subheader("Student-wise Attendance")

            total_sessions = (
                df[["Date", "Session"]]
                .drop_duplicates()
                .shape[0]
            )

            # Count distinct date-session pairs per student
            attended_sessions = (
                df.drop_duplicates(
                    subset=["Student ID", "Date", "Session"]
                )
                .groupby(["Student ID", "Student Name"])
                .size()
                .reset_index(name="Sessions_Attended")
            )

            all_students = pd.DataFrame(
                students,
                columns=["Student ID", "Student Name"]
            )

            student_summary = all_students.merge(
                attended_sessions,
                on=["Student ID", "Student Name"],
                how="left"
            )

            student_summary["Sessions_Attended"] = (
                student_summary["Sessions_Attended"]
                .fillna(0)
                .astype(int)
            )

            student_summary["Attendance Percentage"] = (
                student_summary["Sessions_Attended"]
                / total_sessions * 100
            ).round(2)

            student_summary["Total Recorded Sessions"] = total_sessions

            st.dataframe(
                student_summary,
                width="stretch",
                hide_index=True
            )

            # --------------------------------
            # CSV EXPORT
            # --------------------------------

            st.divider()
            st.subheader("Download Reports")

            csv_data = student_summary.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Student Attendance Report",
                data=csv_data,
                file_name="student_attendance_report.csv",
                mime="text/csv",
                width="content"
            )

            records_csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Complete Attendance Records",
                data=records_csv,
                file_name="complete_attendance_report.csv",
                mime="text/csv",
                width="content"
            )

        else:
            st.info("No attendance records available to generate reports.")

    else:
        st.warning("The attendance database has not been created.")



# ==========================================
# SETTINGS PAGE
# ==========================================
elif page == "Settings":

    st.title("⚙️ Settings")

    st.markdown("### Application Information")

    st.write("**Application:** AttendAI")
    st.write("**Version:** Phase 2 prototype")
    st.write("**Database:** SQLite")
    st.write("**Recognition:** OpenCV with pretrained models")

    st.warning(
        "This is a local development prototype. "
        "Authentication, role permissions, and robust liveness "
        "detection have not yet been implemented."
    )
