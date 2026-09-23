import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CEMS - MAIN LAUNCHER
# ============================================================

st.set_page_config(
    page_title="CEMS",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# SESSION STATE
# ============================================================

if "show_analytics" not in st.session_state:
    st.session_state["show_analytics"] = False


# ============================================================
# LAUNCH NORMAL PYTHON / OPENCV PROGRAM
# ============================================================

def launch_python_script(script_path):
    """
    Launch Python/OpenCV components such as face registration
    and live monitoring in a separate Windows console.
    """

    try:
        full_path = BASE_DIR / script_path

        if not full_path.exists():
            st.error(f"Could not find {script_path}")
            return

        if sys.platform.startswith("win"):
            subprocess.Popen(
                [
                    "cmd.exe",
                    "/k",
                    sys.executable,
                    str(full_path)
                ],
                cwd=str(BASE_DIR)
            )
        else:
            subprocess.Popen(
                [
                    sys.executable,
                    str(full_path)
                ],
                cwd=str(BASE_DIR)
            )

        st.success(
            f"Started {script_path} in a new window."
        )

    except Exception as error:
        st.error(
            f"Could not start {script_path}: {error}"
        )


# ============================================================
# LAUNCH STREAMLIT PAGE
# ============================================================

def launch_streamlit_page(script_path):
    """
    Launch another Streamlit component.
    """

    try:
        full_path = BASE_DIR / script_path

        if not full_path.exists():
            st.error(f"Could not find {script_path}")
            return

        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(full_path)
            ],
            cwd=str(BASE_DIR)
        )

        st.success(
            f"Opened {script_path}"
        )

    except Exception as error:
        st.error(
            f"Could not open {script_path}: {error}"
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🎓 Classroom Engagement Monitoring System"
)

st.write(
    "CEMS provides one place to manage classroom sessions, "
    "register students, monitor engagement and review analytics."
)

st.divider()


# ============================================================
# WORKFLOW
# ============================================================

st.subheader(
    "Lecturer Workflow"
)

st.info(
    "1. Start or select a classroom session  →  "
    "2. Register students if required  →  "
    "3. Start live monitoring  →  "
    "4. View dashboard and analytics"
)

st.divider()


# ============================================================
# MAIN MENU
# ============================================================

st.subheader(
    "CEMS Home"
)

col1, col2 = st.columns(2)


# ============================================================
# LEFT SIDE
# ============================================================

with col1:

    # SESSION MANAGEMENT

    st.markdown(
        "### 📚 Session Management"
    )

    st.write(
        "Start a new classroom session "
        "or review existing sessions."
    )

    if st.button(
        "Start / Select Session",
        use_container_width=True
    ):
        launch_streamlit_page(
            "dashboard/session_management.py"
        )


    # FACE REGISTRATION

    st.markdown(
        "### 📷 Register Student Face"
    )

    st.write(
        "Register a student's face "
        "for recognition during monitoring."
    )

    if st.button(
        "Register Student Face",
        use_container_width=True
    ):
        launch_python_script(
            "face_registration.py"
        )


    # LIVE MONITORING

    st.markdown(
        "### 🎥 Live Monitoring"
    )

    st.write(
        "Start classroom face recognition "
        "and engagement monitoring."
    )

    if st.button(
        "Start Live Monitoring",
        use_container_width=True
    ):
        launch_python_script(
            "tracking.py"
        )


# ============================================================
# RIGHT SIDE
# ============================================================

with col2:

    # DASHBOARD

    st.markdown(
        "### 📊 Dashboard"
    )

    st.write(
        "View classroom engagement records "
        "and dashboard information."
    )

    if st.button(
        "View Dashboard",
        use_container_width=True
    ):
        launch_streamlit_page(
            "dashboard/dashboard.py"
        )


    # ANALYTICS

    st.markdown(
        "### 📈 Analytics"
    )

    st.write(
        "View student engagement, "
        "class averages and low-engagement alerts."
    )

    if st.button(
        "View Analytics",
        use_container_width=True
    ):
        st.session_state["show_analytics"] = True
        st.rerun()


# ============================================================
# ANALYTICS
# ============================================================

if st.session_state["show_analytics"]:

    st.divider()

    st.header(
        "📈 Engagement Analytics"
    )

    st.caption(
        "Simple overview of classroom engagement results."
    )

    try:

        from analytics import get_analytics
        from alerts import get_students_requiring_attention

        analytics_data = get_analytics()


        # ====================================================
        # SUMMARY
        # ====================================================

        average = analytics_data[
            "average_engagement"
        ]

        total_records = analytics_data[
            "total_records"
        ]

        low_count = analytics_data[
            "low_engagement_count"
        ]


        metric1, metric2, metric3 = st.columns(3)


        metric1.metric(
            "Average Engagement",
            f"{average:.1f}%"
        )

        metric2.metric(
            "Total Records",
            total_records
        )

        metric3.metric(
            "Low Engagement",
            low_count
        )


        # ====================================================
        # SIMPLE OVERALL RESULT
        # ====================================================

        st.subheader(
            "📊 Overall Engagement"
        )

        st.write(
            "This shows the average engagement "
            "across the selected classroom records."
        )


        overall_df = pd.DataFrame(
            {
                "Category": [
                    "Average Engagement"
                ],
                "Engagement (%)": [
                    average
                ]
            }
        )

        st.bar_chart(
            overall_df,
            x="Category",
            y="Engagement (%)",
            y_label="Engagement %"
        )

        st.info(
            f"Overall classroom engagement: "
            f"**{average:.1f}%**"
        )


        # ====================================================
        # STUDENT ENGAGEMENT
        # ====================================================

        st.subheader(
            "👥 Student Engagement"
        )

        st.write(
            "This compares the average engagement "
            "of each registered student."
        )


        student_comparison = analytics_data[
            "student_comparison"
        ]


        if student_comparison:

            student_rows = []

            for student_id, score in (
                student_comparison.items()
            ):

                student_rows.append(
                    {
                        "Student": f"Student {student_id}",
                        "Engagement (%)": score
                    }
                )


            student_df = pd.DataFrame(
                student_rows
            )


            st.bar_chart(
                student_df,
                x="Student",
                y="Engagement (%)",
                y_label="Engagement %"
            )


            st.write(
                "**Student Results**"
            )


            for student_id, score in (
                student_comparison.items()
            ):

                if score < 60:

                    st.warning(
                        f"⚠️ Student {student_id}: "
                        f"{score:.1f}%"
                    )

                else:

                    st.success(
                        f"Student {student_id}: "
                        f"{score:.1f}%"
                    )

        else:

            st.info(
                "No registered student "
                "engagement data available."
            )


        # ====================================================
        # DAILY ANALYTICS
        # ====================================================

        daily_averages = analytics_data.get(
            "daily_averages",
            {}
        )


        if daily_averages:

            st.subheader(
                "📅 Daily Average"
            )


            daily_df = pd.DataFrame(
                [
                    {
                        "Date": date,
                        "Average Engagement (%)": score
                    }
                    for date, score
                    in daily_averages.items()
                ]
            )


            st.dataframe(
                daily_df,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # WEEKLY ANALYTICS
        # ====================================================

        weekly_averages = analytics_data.get(
            "weekly_averages",
            {}
        )


        if weekly_averages:

            st.subheader(
                "📆 Weekly Average"
            )


            weekly_df = pd.DataFrame(
                [
                    {
                        "Week": week,
                        "Average Engagement (%)": score
                    }
                    for week, score
                    in weekly_averages.items()
                ]
            )


            st.dataframe(
                weekly_df,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # LOW ENGAGEMENT ALERTS
        # ====================================================

        st.subheader(
            "🚨 Students Requiring Attention"
        )


        attention = (
            get_students_requiring_attention()
        )


        if attention:

            for alert in attention:

                st.warning(
                    alert["message"]
                )

        else:

            st.success(
                "No sustained low-engagement alerts."
            )


        # ====================================================
        # CLOSE ANALYTICS
        # ====================================================

        if st.button(
            "⬅️ Close Analytics"
        ):

            st.session_state[
                "show_analytics"
            ] = False

            st.rerun()


    except Exception as error:

        st.error(
            f"Could not load analytics: {error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CEMS - Classroom Engagement Monitoring System"
)