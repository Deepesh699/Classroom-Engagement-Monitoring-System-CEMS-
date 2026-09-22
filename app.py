import subprocess
import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# CEMS - Main Launcher
# --------------------------------------------------

st.set_page_config(
    page_title="CEMS",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


def launch_python_script(script_path):
    """Launch an existing Python/OpenCV component."""
    try:
        subprocess.Popen(
            [sys.executable, str(BASE_DIR / script_path)],
            cwd=str(BASE_DIR)
        )
        st.success(f"Started {script_path}")
    except Exception as error:
        st.error(f"Could not start {script_path}: {error}")


def launch_streamlit_page(script_path):
    """Launch an existing Streamlit page."""
    try:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(BASE_DIR / script_path)
            ],
            cwd=str(BASE_DIR)
        )
        st.success(f"Opened {script_path}")
    except Exception as error:
        st.error(f"Could not open {script_path}: {error}")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎓 Classroom Engagement Monitoring System")

st.write(
    "CEMS provides one place to manage classroom sessions, "
    "register students, monitor engagement and review analytics."
)

st.divider()


# --------------------------------------------------
# WORKFLOW
# --------------------------------------------------

st.subheader("Lecturer Workflow")

st.info(
    "1. Start or select a classroom session  →  "
    "2. Register students if required  →  "
    "3. Start live monitoring  →  "
    "4. View dashboard and analytics"
)

st.divider()


# --------------------------------------------------
# MAIN MENU
# --------------------------------------------------

st.subheader("CEMS Home")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 📚 Session Management")
    st.write(
        "Start a new classroom session or review existing sessions."
    )

    if st.button(
        "Start / Select Session",
        use_container_width=True
    ):
        launch_streamlit_page(
            "dashboard/session_management.py"
        )

    st.markdown("### 📷 Register Student Face")
    st.write(
        "Register a student's face for recognition during monitoring."
    )

    if st.button(
        "Register Student Face",
        use_container_width=True
    ):
        launch_python_script(
            "face_registration.py"
        )

    st.markdown("### 🎥 Live Monitoring")
    st.write(
        "Start classroom face recognition and engagement monitoring."
    )

    if st.button(
        "Start Live Monitoring",
        use_container_width=True
    ):
        launch_python_script(
            "tracking.py"
        )


with col2:

    st.markdown("### 📊 Dashboard")
    st.write(
        "View classroom engagement records and dashboard information."
    )

    if st.button(
        "View Dashboard",
        use_container_width=True
    ):
        launch_streamlit_page(
            "dashboard/dashboard.py"
        )

    st.markdown("### 📈 Analytics")
    st.write(
        "View student and session engagement analytics."
    )

    if st.button(
        "View Analytics",
        use_container_width=True
    ):
        st.session_state["show_analytics"] = True


# --------------------------------------------------
# ANALYTICS SUMMARY
# --------------------------------------------------

if st.session_state.get("show_analytics", False):

    st.divider()

    st.header("📈 Engagement Analytics")

    try:
        from analytics import get_analytics
        from alerts import get_students_requiring_attention

        analytics_data = get_analytics()

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Average Engagement",
            f"{analytics_data['average_engagement']}%"
        )

        metric2.metric(
            "Total Records",
            analytics_data["total_records"]
        )

        metric3.metric(
            "Low Engagement Records",
            analytics_data["low_engagement_count"]
        )

        st.subheader("Student Average Engagement")

        student_comparison = analytics_data[
            "student_comparison"
        ]

        if student_comparison:
            st.bar_chart(student_comparison)
        else:
            st.info("No registered student engagement data available.")

        st.subheader("Session Average Engagement")

        session_comparison = analytics_data[
            "session_comparison"
        ]

        if session_comparison:
            st.bar_chart(session_comparison)
        else:
            st.info("No session engagement data available.")

        st.subheader("Students Requiring Attention")

        attention = get_students_requiring_attention()

        if attention:
            for alert in attention:
                st.warning(alert["message"])
        else:
            st.success(
                "No sustained low-engagement alerts."
            )

        if st.button("Close Analytics"):
            st.session_state["show_analytics"] = False
            st.rerun()

    except Exception as error:
        st.error(
            f"Could not load analytics: {error}"
        )


st.divider()

st.caption(
    "CEMS - Classroom Engagement Monitoring System"
)