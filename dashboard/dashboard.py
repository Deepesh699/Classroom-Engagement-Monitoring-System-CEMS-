import os
import sys
import pandas as pd
import streamlit as st

# ============================================================
# CONNECT DASHBOARD TO PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CEMS Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA FROM SQLITE DATABASE
# ============================================================

def load_data():
    """
    Load engagement records from the SQLite database.
    """

    try:
        records = database.get_all_records()

        if not records:
            return pd.DataFrame()

        # database.get_all_records() returns:
        # id
        # timestamp
        # student_id
        # engagement_score
        # status
        # registered_student_id
        # session_id
        # orientation

        data = pd.DataFrame(
            records,
            columns=[
                "id",
                "timestamp",
                "student_id",
                "engagement_score",
                "status",
                "registered_student_id",
                "session_id",
                "orientation"
            ]
        )

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

        data["engagement_score"] = pd.to_numeric(
            data["engagement_score"],
            errors="coerce"
        )

        return data

    except Exception as e:
        st.error(f"Unable to load engagement data: {e}")
        return pd.DataFrame()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_students_safe():
    try:
        return database.get_students()
    except Exception:
        return []


def get_sessions_safe():
    try:
        return database.get_sessions()
    except Exception:
        return []


def get_units_safe():
    try:
        return database.get_units()
    except Exception:
        return []


def get_classrooms_safe():
    try:
        return database.get_classrooms()
    except Exception:
        return []


def get_active_session_safe():
    try:
        return database.get_active_session_id()
    except Exception:
        return None


def status_from_score(score):
    if score >= 75:
        return "Engaged"
    elif score >= 50:
        return "Moderate"
    else:
        return "Low"


# ============================================================
# LOAD DATABASE DATA
# ============================================================

data = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 CEMS")
st.sidebar.caption("Classroom Engagement Monitoring System")

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live Classroom",
        "Student History",
        "Weekly Analytics",
        "Classroom Comparison",
        "Session History"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

active_session = get_active_session_safe()

if active_session is not None:
    st.sidebar.success(f"Active Session: {active_session}")
else:
    st.sidebar.info("No active session")


# ============================================================
# HEADER
# ============================================================

st.title("📊 CEMS Dashboard")
st.caption("Classroom Engagement Monitoring System")


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.header("Dashboard Overview")

    if data.empty:

        st.warning("No engagement data is currently available.")

        st.info(
            "The dashboard is ready, but there are no "
            "engagement records to display yet."
        )

    else:

        # ----------------------------
        # KPIs
        # ----------------------------

        total_records = len(data)

        average_engagement = data["engagement_score"].mean()

        engaged_count = len(
            data[data["engagement_score"] >= 75]
        )

        low_engagement_count = len(
            data[data["engagement_score"] < 50]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Records",
            total_records
        )

        col2.metric(
            "Average Engagement",
            f"{average_engagement:.1f}%"
        )

        col3.metric(
            "Engaged Records",
            engaged_count
        )

        col4.metric(
            "Low Engagement",
            low_engagement_count
        )

        st.divider()

        # ----------------------------
        # ENGAGEMENT TREND
        # ----------------------------

        st.subheader("Engagement Trend")

        trend_data = (
            data.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .set_index("timestamp")
        )

        if not trend_data.empty:
            st.line_chart(
                trend_data["engagement_score"]
            )
        else:
            st.info("No valid timestamp data available.")

        # ----------------------------
        # STATUS DISTRIBUTION
        # ----------------------------

        st.subheader("Engagement Status")

        status_counts = (
            data["status"]
            .fillna("Unknown")
            .value_counts()
        )

        if not status_counts.empty:
            st.bar_chart(status_counts)

        # ----------------------------
        # RECENT RECORDS
        # ----------------------------

        st.subheader("Recent Engagement Records")

        display_columns = [
            "timestamp",
            "student_id",
            "engagement_score",
            "status",
            "session_id",
            "orientation"
        ]

        st.dataframe(
            data[display_columns]
            .sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# LIVE CLASSROOM PAGE
# ============================================================

elif page == "Live Classroom":

    st.header("Live Classroom")

    active_session = get_active_session_safe()

    if active_session is None:

        st.warning("There is currently no active classroom session.")

    else:

        st.success(
            f"Session {active_session} is currently active."
        )

        if data.empty:

            st.info(
                "The session is active but no engagement "
                "records have been recorded yet."
            )

        else:

            session_data = data[
                data["session_id"] == active_session
            ].copy()

            if session_data.empty:

                st.info(
                    "No engagement records have been recorded "
                    "for the active session."
                )

            else:

                avg_score = session_data[
                    "engagement_score"
                ].mean()

                low_count = len(
                    session_data[
                        session_data["engagement_score"] < 50
                    ]
                )

                unique_students = (
                    session_data["student_id"]
                    .nunique()
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Average Engagement",
                    f"{avg_score:.1f}%"
                )

                col2.metric(
                    "Students Detected",
                    unique_students
                )

                col3.metric(
                    "Low Engagement",
                    low_count
                )

                st.subheader("Live Engagement Records")

                st.dataframe(
                    session_data[
                        [
                            "timestamp",
                            "student_id",
                            "engagement_score",
                            "status",
                            "orientation"
                        ]
                    ].sort_values(
                        "timestamp",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                chart_data = (
                    session_data
                    .dropna(subset=["timestamp"])
                    .sort_values("timestamp")
                    .set_index("timestamp")
                )

                if not chart_data.empty:

                    st.subheader("Live Engagement Trend")

                    st.line_chart(
                        chart_data["engagement_score"]
                    )


# ============================================================
# STUDENT HISTORY PAGE
# ============================================================

elif page == "Student History":

    st.header("Student History")

    if data.empty:

        st.warning("No student engagement history is available.")

    else:

        students = sorted(
            data["student_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_student = st.selectbox(
            "Select Student",
            students
        )

        student_data = data[
            data["student_id"].astype(str)
            == selected_student
        ].copy()

        average_score = (
            student_data["engagement_score"].mean()
        )

        highest_score = (
            student_data["engagement_score"].max()
        )

        lowest_score = (
            student_data["engagement_score"].min()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Engagement",
            f"{average_score:.1f}%"
        )

        col2.metric(
            "Highest Score",
            f"{highest_score:.0f}%"
        )

        col3.metric(
            "Lowest Score",
            f"{lowest_score:.0f}%"
        )

        st.subheader("Student Engagement Trend")

        student_chart = (
            student_data
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .set_index("timestamp")
        )

        if not student_chart.empty:
            st.line_chart(
                student_chart["engagement_score"]
            )

        st.subheader("Student Records")

        st.dataframe(
            student_data[
                [
                    "timestamp",
                    "student_id",
                    "engagement_score",
                    "status",
                    "session_id",
                    "orientation"
                ]
            ].sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# WEEKLY ANALYTICS PAGE
# ============================================================

elif page == "Weekly Analytics":

    st.header("Weekly Analytics")

    if data.empty:

        st.warning("No engagement data is available.")

    else:

        weekly_data = data.dropna(
            subset=["timestamp"]
        ).copy()

        if weekly_data.empty:

            st.info("No valid timestamp information is available.")

        else:

            weekly_data["week"] = (
                weekly_data["timestamp"]
                .dt.to_period("W")
                .astype(str)
            )

            weekly_summary = (
                weekly_data
                .groupby("week")["engagement_score"]
                .agg(
                    Average="mean",
                    Minimum="min",
                    Maximum="max",
                    Records="count"
                )
                .reset_index()
            )

            st.subheader("Average Engagement by Week")

            weekly_chart = weekly_summary.set_index(
                "week"
            )

            st.line_chart(
                weekly_chart["Average"]
            )

            st.subheader("Weekly Summary")

            st.dataframe(
                weekly_summary,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# CLASSROOM COMPARISON PAGE
# ============================================================

elif page == "Classroom Comparison":

    st.header("Classroom Comparison")

    sessions = get_sessions_safe()
    classrooms = get_classrooms_safe()

    if not sessions:

        st.warning("No classroom sessions are available.")

    elif data.empty:

        st.warning(
            "Sessions exist, but there is no engagement data "
            "available for comparison."
        )

    else:

        st.subheader("Engagement by Session")

        comparison = (
            data.groupby("session_id")["engagement_score"]
            .agg(
                Average="mean",
                Minimum="min",
                Maximum="max",
                Records="count"
            )
            .reset_index()
        )

        comparison["Average"] = (
            comparison["Average"].round(2)
        )

        st.bar_chart(
            comparison.set_index("session_id")["Average"]
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        if classrooms:

            st.subheader("Registered Classrooms")

            classroom_df = pd.DataFrame(
                classrooms
            )

            st.dataframe(
                classroom_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# SESSION HISTORY PAGE
# ============================================================

elif page == "Session History":

    st.header("Session History")

    sessions = get_sessions_safe()

    if not sessions:

        st.warning("No sessions have been created.")

    else:

        session_rows = []

        for session in sessions:

            session_id = session[0]

            session_records = pd.DataFrame()

            if not data.empty:
                session_records = data[
                    data["session_id"] == session_id
                ]

            if session_records.empty:
                avg_engagement = 0
                record_count = 0
            else:
                avg_engagement = (
                    session_records[
                        "engagement_score"
                    ].mean()
                )

                record_count = len(session_records)

            session_rows.append(
                {
                    "Session ID": session_id,
                    "Unit": (
                        session[1]
                        if len(session) > 1
                        else ""
                    ),
                    "Unit Name": (
                        session[2]
                        if len(session) > 2
                        else ""
                    ),
                    "Classroom": (
                        session[3]
                        if len(session) > 3
                        else ""
                    ),
                    "Started": (
                        session[4]
                        if len(session) > 4
                        else ""
                    ),
                    "Ended": (
                        session[5]
                        if len(session) > 5
                        else ""
                    ),
                    "Records": record_count,
                    "Average Engagement": round(
                        avg_engagement,
                        2
                    )
                }
            )

        session_df = pd.DataFrame(session_rows)

        st.dataframe(
            session_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CEMS — Classroom Engagement Monitoring System | "
    "SQLite + Python + Streamlit"
)