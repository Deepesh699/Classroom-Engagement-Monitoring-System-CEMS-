import os
import sys
import sqlite3
import subprocess
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database

from analytics import (
    get_analytics,
    get_daily_engagement_averages,
    get_weekly_engagement_averages,
    get_student_average,
)

from alerts import check_sustained_low_engagement


DB_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cems.db"
)

LOW_THRESHOLD = 60


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CEMS Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def load_students():
    """Load registered students."""

    try:
        conn = get_connection()

        df = pd.read_sql_query(
            """
            SELECT
                id AS registered_student_id,
                student_number,
                student_name,
                created_at
            FROM students
            ORDER BY student_name
            """,
            conn
        )

        conn.close()
        return df

    except Exception:

        return pd.DataFrame(
            columns=[
                "registered_student_id",
                "student_number",
                "student_name",
                "created_at"
            ]
        )


def load_sessions():
    """Load sessions with unit and classroom information."""

    try:

        conn = get_connection()

        df = pd.read_sql_query(
            """
            SELECT
                s.id AS session_id,
                s.unit_id,
                u.unit_code,
                u.unit_name,
                s.classroom_id,
                c.classroom_name,
                s.start_time,
                s.end_time
            FROM sessions s
            LEFT JOIN units u
                ON s.unit_id = u.id
            LEFT JOIN classrooms c
                ON s.classroom_id = c.id
            ORDER BY s.start_time DESC
            """,
            conn
        )

        conn.close()

        if not df.empty:

            df["start_time"] = pd.to_datetime(
                df["start_time"],
                errors="coerce"
            )

            df["end_time"] = pd.to_datetime(
                df["end_time"],
                errors="coerce"
            )

        return df

    except Exception:

        return pd.DataFrame()


def load_data():
    """
    Load engagement records.

    IMPORTANT:
    engagement_records.student_id
        = temporary tracker ID

    engagement_records.registered_student_id
        = real registered student ID
    """

    try:

        conn = get_connection()

        query = """
        SELECT
            er.id,
            er.timestamp,
            er.student_id AS track_id,
            er.engagement_score,
            er.status,
            er.registered_student_id,
            er.session_id,
            er.orientation,
            s.student_number,
            s.student_name
        FROM engagement_records er
        LEFT JOIN students s
            ON er.registered_student_id = s.id
        ORDER BY er.timestamp DESC
        """

        data = pd.read_sql_query(
            query,
            conn
        )

        conn.close()

        if data.empty:
            return data

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

        data["engagement_score"] = pd.to_numeric(
            data["engagement_score"],
            errors="coerce"
        )

        data["student_name"] = (
            data["student_name"]
            .fillna("Unknown")
        )

        data["student_number"] = (
            data["student_number"]
            .fillna("Unknown")
        )

        data["orientation"] = (
            data["orientation"]
            .fillna("Unknown")
        )

        data["status"] = (
            data["status"]
            .fillna("Unknown")
        )

        # Ignore old incomplete/collecting states
        data = data[
            ~data["status"]
            .astype(str)
            .str.lower()
            .str.contains(
                "collecting",
                na=False
            )
        ].copy()

        return data

    except Exception as e:

        st.error(
            f"Unable to load engagement data: {e}"
        )

        return pd.DataFrame()


def get_active_session_safe():

    try:
        return database.get_active_session_id()

    except Exception:
        return None


# ============================================================
# LIVE MONITORING LAUNCHER
# ============================================================

def launch_live_monitoring():
    """
    Launch tracking.py in a separate terminal.

    The existing tracking.py can then ask for:
    1 - USB Camera
    2 - CCTV / IP Camera
    3 - Video File
    """

    tracking_script = os.path.join(
        PROJECT_ROOT,
        "tracking.py"
    )

    venv_python = os.path.join(
        PROJECT_ROOT,
        ".venv",
        "Scripts",
        "python.exe"
    )

    python_executable = (
        venv_python
        if os.path.exists(venv_python)
        else sys.executable
    )

    if not os.path.exists(tracking_script):

        st.error(
            "tracking.py could not be found."
        )

        return

    try:

        if os.name == "nt":

            subprocess.Popen(
                [
                    "cmd.exe",
                    "/k",
                    python_executable,
                    tracking_script
                ],
                cwd=PROJECT_ROOT,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

        else:

            subprocess.Popen(
                [
                    python_executable,
                    tracking_script
                ],
                cwd=PROJECT_ROOT
            )

        st.success(
            "Live monitoring launched."
        )

    except Exception as e:

        st.error(
            f"Unable to launch live monitoring: {e}"
        )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def student_label(row):

    name = row.get(
        "student_name",
        "Unknown"
    )

    number = row.get(
        "student_number",
        "Unknown"
    )

    return f"{name} - {number}"


def session_label(row):

    session_id = row.get(
        "session_id",
        ""
    )

    unit = row.get(
        "unit_code",
        ""
    )

    classroom = row.get(
        "classroom_name",
        ""
    )

    start = row.get(
        "start_time"
    )

    if pd.notna(start):

        date_text = start.strftime(
            "%d %b %Y"
        )

        time_text = start.strftime(
            "%H:%M"
        )

    else:

        date_text = "Unknown date"
        time_text = ""

    return (
        f"{unit} | {classroom} | "
        f"{date_text} | {time_text} | "
        f"Session {session_id}"
    )


def display_name(value):

    if pd.isna(value):
        return "Unknown"

    return str(value)


# ============================================================
# LOAD DATA
# ============================================================

data = load_data()

students_df = load_students()

sessions_df = load_sessions()

active_session = get_active_session_safe()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 CEMS"
)

st.sidebar.caption(
    "Classroom Engagement Monitoring System"
)

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live Classroom",
        "Student History",
        "Weekly Analytics",
        "Session History"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()


if active_session is not None:

    st.sidebar.success(
        f"Active Session: {active_session}"
    )

else:

    st.sidebar.info(
        "No active session"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 CEMS Dashboard"
)

st.caption(
    "Classroom Engagement Monitoring System"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header(
        "Dashboard Overview"
    )

    total_registered = len(
        students_df
    )


    # --------------------------------------------------------
    # CURRENT / LATEST SESSION
    # --------------------------------------------------------

    if active_session is not None:

        latest_session = int(
            active_session
        )

    elif (
        not data.empty
        and data["session_id"]
        .notna()
        .any()
    ):

        latest_session = int(
            data[
                "session_id"
            ]
            .dropna()
            .max()
        )

    else:

        latest_session = None


    # --------------------------------------------------------
    # FILTER OVERVIEW TO CURRENT / LATEST SESSION
    # --------------------------------------------------------

    if (
        latest_session is not None
        and not data.empty
    ):

        overview_data = data[
            data["session_id"]
            == latest_session
        ].copy()

    else:

        overview_data = pd.DataFrame(
            columns=data.columns
        )


    if latest_session is not None:

        overview_analytics = (
            get_analytics(
                latest_session
            )
        )

    else:

        overview_analytics = None


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    if overview_data.empty:

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        col1.metric(
            "Registered Students",
            total_registered
        )

        col2.metric(
            "Average Engagement",
            "N/A"
        )

        col3.metric(
            "Engaged Records",
            0
        )

        col4.metric(
            "Low Engagement",
            0
        )

        col5.metric(
            "Latest Session",
            (
                latest_session
                if latest_session is not None
                else "None"
            )
        )

        st.info(
            "No engagement records are currently "
            "available for this session."
        )

    else:

        average_engagement = (
            overview_analytics[
                "average_engagement"
            ]
        )

        engaged_count = len(
            overview_data[
                overview_data[
                    "status"
                ]
                .astype(str)
                .str.lower()
                .eq("engaged")
            ]
        )

        low_count = (
            overview_analytics[
                "low_engagement_count"
            ]
        )


        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        col1.metric(
            "Registered Students",
            total_registered
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
            "Low Engagement Records",
            low_count
        )

        col5.metric(
            "Current / Latest Session",
            latest_session
        )

        st.divider()


        # ----------------------------------------------------
        # ENGAGEMENT TREND
        # ----------------------------------------------------

        st.subheader(
            "📈 Engagement Trend"
        )

        trend = (
            overview_data
            .dropna(
                subset=[
                    "timestamp",
                    "engagement_score"
                ]
            )
            .sort_values(
                "timestamp"
            )
        )

        if trend.empty:

            st.info(
                "No valid engagement trend "
                "data available."
            )

        else:

            trend_chart = (
                trend[
                    [
                        "timestamp",
                        "engagement_score"
                    ]
                ]
                .set_index(
                    "timestamp"
                )
            )

            st.line_chart(
                trend_chart,
                use_container_width=True
            )


        # ----------------------------------------------------
        # RECENT RECORDS
        # ----------------------------------------------------

        st.subheader(
            "🧑‍🎓 Recent Engagement Records"
        )

        recent = (
            overview_data
            .sort_values(
                "timestamp",
                ascending=False
            )
            .head(20)
            .copy()
        )

        recent_display = recent[
            [
                "timestamp",
                "student_name",
                "student_number",
                "engagement_score",
                "status",
                "orientation",
                "session_id"
            ]
        ].copy()

        recent_display.columns = [
            "Time",
            "Student Name",
            "Student Number",
            "Score",
            "Status",
            "Orientation",
            "Session"
        ]

        st.dataframe(
            recent_display,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # STUDENTS REQUIRING ATTENTION
        # ----------------------------------------------------

        st.subheader(
            "🚨 Students Requiring Attention"
        )

        recognised_students = (
            overview_data[
                overview_data[
                    "registered_student_id"
                ].notna()
            ][
                [
                    "registered_student_id",
                    "student_name",
                    "student_number"
                ]
            ]
            .drop_duplicates()
        )


        sustained_alerts = []


        for _, student in (
            recognised_students.iterrows()
        ):

            registered_id = int(
                student[
                    "registered_student_id"
                ]
            )


            # IMPORTANT:
            # check_sustained_low_engagement()
            # returns a dictionary:
            #
            # {
            #     "alert": True/False,
            #     "message": "..."
            # }
            #
            # Therefore we must check the
            # "alert" value, not the dictionary itself.

            alert_result = (
                check_sustained_low_engagement(
                    registered_id,
                    session_id=latest_session,
                    threshold=LOW_THRESHOLD,
                    required_count=3
                )
            )


            if alert_result.get(
                "alert",
                False
            ):

                sustained_alerts.append(
                    {
                        "student": student,
                        "message": alert_result.get(
                            "message",
                            ""
                        )
                    }
                )


        if not sustained_alerts:

            st.success(
                "No sustained low-engagement "
                "students detected in this session."
            )

        else:

            for alert_item in sustained_alerts:

                student = (
                    alert_item[
                        "student"
                    ]
                )

                backend_message = (
                    alert_item[
                        "message"
                    ]
                )

                if backend_message:

                    st.warning(
                        f"⚠️ "
                        f"{student['student_name']} "
                        f"({student['student_number']}) — "
                        f"{backend_message}"
                    )

                else:

                    st.warning(
                        f"⚠️ "
                        f"{student['student_name']} "
                        f"({student['student_number']}) — "
                        f"sustained low engagement "
                        f"detected in Session "
                        f"{latest_session}."
                    )


# ============================================================
# LIVE CLASSROOM
# ============================================================

elif page == "Live Classroom":

    st.header(
        "Live Classroom"
    )


    # --------------------------------------------------------
    # START LIVE MONITORING
    # --------------------------------------------------------

    monitor_col1, monitor_col2 = (
        st.columns(
            [1, 4]
        )
    )


    with monitor_col1:

        if st.button(
            "🎥 Start Live Monitoring",
            use_container_width=True,
            disabled=(
                active_session is None
            )
        ):

            launch_live_monitoring()


    with monitor_col2:

        if active_session is None:

            st.caption(
                "Start a classroom session "
                "before opening live monitoring."
            )

        else:

            st.caption(
                f"Monitoring will save engagement "
                f"records to Session "
                f"{active_session}."
            )


    # --------------------------------------------------------
    # ACTIVE SESSION
    # --------------------------------------------------------

    if active_session is None:

        st.warning(
            "There is currently no active "
            "classroom session."
        )

    else:

        st.success(
            f"Session {active_session} "
            f"is currently active."
        )


        if data.empty:

            st.info(
                "The session is active but no "
                "engagement records have been "
                "recorded yet."
            )

        else:

            session_data = data[
                data["session_id"]
                == active_session
            ].copy()


            if session_data.empty:

                st.info(
                    "No engagement records are "
                    "available for the active session."
                )

            else:

                # Latest record for each REAL
                # registered student

                recognised = session_data[
                    session_data[
                        "registered_student_id"
                    ].notna()
                ].copy()


                latest_students = (
                    recognised
                    .sort_values(
                        "timestamp"
                    )
                    .groupby(
                        "registered_student_id",
                        as_index=False
                    )
                    .tail(1)
                )


                class_average = (
                    latest_students[
                        "engagement_score"
                    ].mean()
                    if not latest_students.empty
                    else 0
                )


                engaged = len(
                    latest_students[
                        latest_students[
                            "status"
                        ]
                        .astype(str)
                        .str.lower()
                        .eq("engaged")
                    ]
                )


                neutral = len(
                    latest_students[
                        latest_students[
                            "status"
                        ]
                        .astype(str)
                        .str.lower()
                        .isin(
                            [
                                "neutral",
                                "moderate"
                            ]
                        )
                    ]
                )


                low = len(
                    latest_students[
                        latest_students[
                            "engagement_score"
                        ] < LOW_THRESHOLD
                    ]
                )


                col1, col2, col3, col4 = (
                    st.columns(4)
                )


                col1.metric(
                    "Live Class Average",
                    f"{class_average:.1f}%"
                )

                col2.metric(
                    "Engaged",
                    engaged
                )

                col3.metric(
                    "Neutral",
                    neutral
                )

                col4.metric(
                    "Low Engagement",
                    low
                )


                st.divider()


                # ------------------------------------------------
                # RECOGNISED STUDENTS
                # ------------------------------------------------

                st.subheader(
                    "Recognised Students"
                )


                if latest_students.empty:

                    st.info(
                        "No recognised registered "
                        "students are currently available."
                    )

                else:

                    live_display = (
                        latest_students[
                            [
                                "student_name",
                                "student_number",
                                "engagement_score",
                                "status",
                                "orientation",
                                "timestamp"
                            ]
                        ]
                        .copy()
                    )


                    live_display[
                        "timestamp"
                    ] = (
                        live_display[
                            "timestamp"
                        ]
                        .dt
                        .strftime(
                            "%H:%M:%S"
                        )
                    )


                    live_display.columns = [
                        "Student Name",
                        "Student Number",
                        "Engagement Score",
                        "Status",
                        "Orientation",
                        "Last Updated"
                    ]


                    st.dataframe(
                        live_display,
                        use_container_width=True,
                        hide_index=True
                    )


                # ------------------------------------------------
                # LIVE ENGAGEMENT TREND
                # ------------------------------------------------

                st.subheader(
                    "Live Engagement Trend"
                )


                chart_data = (
                    session_data
                    .dropna(
                        subset=[
                            "timestamp",
                            "engagement_score"
                        ]
                    )
                    .sort_values(
                        "timestamp"
                    )
                    .set_index(
                        "timestamp"
                    )
                )


                if not chart_data.empty:

                    st.line_chart(
                        chart_data[
                            "engagement_score"
                        ],
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No trend data is available."
                    )


# ============================================================
# STUDENT HISTORY
# ============================================================

elif page == "Student History":

    st.header(
        "Student History"
    )


    if students_df.empty:

        st.warning(
            "No registered students "
            "are available."
        )

    else:

        selectable_students = (
            students_df[
                [
                    "registered_student_id",
                    "student_name",
                    "student_number"
                ]
            ]
            .copy()
        )


        selectable_students[
            "label"
        ] = (
            selectable_students
            .apply(
                student_label,
                axis=1
            )
        )


        labels = (
            selectable_students[
                "label"
            ]
            .tolist()
        )


        selected_label = st.selectbox(
            "Select Student",
            labels
        )


        selected_row = (
            selectable_students[
                selectable_students[
                    "label"
                ]
                == selected_label
            ]
            .iloc[0]
        )


        selected_id = (
            selected_row[
                "registered_student_id"
            ]
        )


        if not data.empty:

            student_data = data[
                data[
                    "registered_student_id"
                ]
                == selected_id
            ].copy()

        else:

            student_data = pd.DataFrame()


        st.caption(
            f"Registered Student ID: "
            f"{selected_id}"
        )


        if student_data.empty:

            st.info(
                "No engagement records are "
                "available for this student."
            )

        else:

            # Uses analytics.py

            average_score = (
                get_student_average(
                    int(selected_id)
                )
            )


            record_count = len(
                student_data
            )


            low_count = len(
                student_data[
                    student_data[
                        "engagement_score"
                    ] < LOW_THRESHOLD
                ]
            )


            session_count = (
                student_data[
                    "session_id"
                ]
                .dropna()
                .nunique()
            )


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            col1.metric(
                "Average Engagement",
                f"{average_score:.1f}%"
            )

            col2.metric(
                "Number of Records",
                record_count
            )

            col3.metric(
                "Low Engagement Count",
                low_count
            )

            col4.metric(
                "Sessions",
                session_count
            )


            st.divider()


            # ------------------------------------------------
            # ENGAGEMENT TREND
            # ------------------------------------------------

            st.subheader(
                "📈 Engagement Trend"
            )


            student_chart = (
                student_data
                .dropna(
                    subset=[
                        "timestamp",
                        "engagement_score"
                    ]
                )
                .sort_values(
                    "timestamp"
                )
                .set_index(
                    "timestamp"
                )
            )


            if not student_chart.empty:

                st.line_chart(
                    student_chart[
                        "engagement_score"
                    ],
                    use_container_width=True
                )


            # ------------------------------------------------
            # STATUS HISTORY
            # ------------------------------------------------

            col1, col2 = st.columns(2)


            with col1:

                st.subheader(
                    "Status History"
                )

                status_counts = (
                    student_data[
                        "status"
                    ]
                    .fillna(
                        "Unknown"
                    )
                    .value_counts()
                )

                st.bar_chart(
                    status_counts
                )


            # ------------------------------------------------
            # ORIENTATION HISTORY
            # ------------------------------------------------

            with col2:

                st.subheader(
                    "Orientation History"
                )

                orientation_counts = (
                    student_data[
                        "orientation"
                    ]
                    .fillna(
                        "Unknown"
                    )
                    .value_counts()
                )

                st.bar_chart(
                    orientation_counts
                )


            # ------------------------------------------------
            # SESSION HISTORY
            # ------------------------------------------------

            st.subheader(
                "Session History"
            )


            session_summary = (
                student_data
                .groupby(
                    "session_id",
                    dropna=False
                )[
                    "engagement_score"
                ]
                .agg(
                    Average="mean",
                    Records="count",
                    Minimum="min",
                    Maximum="max"
                )
                .reset_index()
            )


            session_summary[
                "Average"
            ] = (
                session_summary[
                    "Average"
                ]
                .round(1)
            )


            st.dataframe(
                session_summary,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # ALL STUDENT RECORDS
            # ------------------------------------------------

            st.subheader(
                "All Student Records"
            )


            student_display = (
                student_data[
                    [
                        "timestamp",
                        "engagement_score",
                        "status",
                        "orientation",
                        "session_id",
                        "track_id"
                    ]
                ]
                .copy()
            )


            student_display.columns = [
                "Time",
                "Score",
                "Status",
                "Orientation",
                "Session",
                "Track ID"
            ]


            st.dataframe(
                student_display
                .sort_values(
                    "Time",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# WEEKLY ANALYTICS
# ============================================================

elif page == "Weekly Analytics":

    st.header(
        "Weekly Analytics"
    )


    if data.empty:

        st.warning(
            "No engagement data is available."
        )

    else:

        analytics_data = (
            data
            .dropna(
                subset=[
                    "timestamp",
                    "engagement_score"
                ]
            )
            .copy()
        )


        if analytics_data.empty:

            st.info(
                "No valid analytics data "
                "is available."
            )

        else:

            # ------------------------------------------------
            # OVERALL TREND
            # ------------------------------------------------

            st.subheader(
                "Average Engagement Over Time"
            )


            trend = (
                analytics_data
                .sort_values(
                    "timestamp"
                )
                .set_index(
                    "timestamp"
                )
            )


            st.line_chart(
                trend[
                    "engagement_score"
                ],
                use_container_width=True
            )


            # ------------------------------------------------
            # STATUS DISTRIBUTION
            # ------------------------------------------------

            st.subheader(
                "Engagement Status Distribution"
            )


            status_distribution = (
                analytics_data[
                    "status"
                ]
                .fillna(
                    "Unknown"
                )
                .value_counts()
            )


            st.bar_chart(
                status_distribution
            )


            # ------------------------------------------------
            # STUDENT COMPARISON
            # ------------------------------------------------

            st.subheader(
                "Student Average Comparison"
            )


            recognised = (
                analytics_data[
                    analytics_data[
                        "registered_student_id"
                    ].notna()
                ]
                .copy()
            )


            if recognised.empty:

                st.info(
                    "No recognised student "
                    "data available."
                )

            else:

                student_average = (
                    recognised
                    .groupby(
                        [
                            "registered_student_id",
                            "student_name",
                            "student_number"
                        ]
                    )[
                        "engagement_score"
                    ]
                    .mean()
                    .reset_index()
                )


                student_average[
                    "Student"
                ] = (
                    student_average[
                        "student_name"
                    ]
                    + " - "
                    + student_average[
                        "student_number"
                    ]
                )


                student_average[
                    "Average"
                ] = (
                    student_average[
                        "engagement_score"
                    ]
                    .round(1)
                )


                st.bar_chart(
                    student_average
                    .set_index(
                        "Student"
                    )[
                        "Average"
                    ]
                )


            # ------------------------------------------------
            # DAILY AVERAGE
            # analytics.py
            # ------------------------------------------------

            st.subheader(
                "Daily Average"
            )


            daily_values = (
                get_daily_engagement_averages()
            )


            if daily_values:

                daily = pd.DataFrame(
                    list(
                        daily_values.items()
                    ),
                    columns=[
                        "date",
                        "Average"
                    ]
                )


                daily[
                    "Average"
                ] = (
                    pd.to_numeric(
                        daily[
                            "Average"
                        ],
                        errors="coerce"
                    )
                    .round(1)
                )


                st.line_chart(
                    daily
                    .set_index(
                        "date"
                    )[
                        "Average"
                    ]
                )

            else:

                st.info(
                    "No daily analytics "
                    "are available."
                )


            # ------------------------------------------------
            # WEEKLY AVERAGE
            # analytics.py
            # ------------------------------------------------

            st.subheader(
                "Weekly Average"
            )


            weekly_values = (
                get_weekly_engagement_averages()
            )


            if weekly_values:

                weekly = pd.DataFrame(
                    list(
                        weekly_values.items()
                    ),
                    columns=[
                        "week",
                        "Average"
                    ]
                )


                weekly[
                    "Average"
                ] = (
                    pd.to_numeric(
                        weekly[
                            "Average"
                        ],
                        errors="coerce"
                    )
                    .round(1)
                )


                st.line_chart(
                    weekly
                    .set_index(
                        "week"
                    )[
                        "Average"
                    ]
                )


                st.dataframe(
                    weekly,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No weekly analytics "
                    "are available."
                )


# ============================================================
# SESSION HISTORY
# ============================================================

elif page == "Session History":

    st.header(
        "Session History"
    )


    if sessions_df.empty:

        st.warning(
            "No sessions have been created."
        )

    else:

        session_options = {}


        for _, row in (
            sessions_df.iterrows()
        ):

            label = session_label(
                row
            )

            session_options[
                label
            ] = row[
                "session_id"
            ]


        selected_session_label = (
            st.selectbox(
                "Select Session",
                list(
                    session_options.keys()
                )
            )
        )


        selected_session_id = (
            session_options[
                selected_session_label
            ]
        )


        selected_session = (
            sessions_df[
                sessions_df[
                    "session_id"
                ]
                == selected_session_id
            ]
            .iloc[0]
        )


        # ----------------------------------------------------
        # SESSION INFORMATION
        # ----------------------------------------------------

        st.subheader(
            "Session Information"
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        col1.metric(
            "Unit",
            display_name(
                selected_session[
                    "unit_code"
                ]
            )
        )


        col2.metric(
            "Classroom",
            display_name(
                selected_session[
                    "classroom_name"
                ]
            )
        )


        start_time = (
            selected_session[
                "start_time"
            ]
        )

        end_time = (
            selected_session[
                "end_time"
            ]
        )


        col3.metric(
            "Start Time",
            (
                start_time.strftime(
                    "%d %b %Y %H:%M"
                )
                if pd.notna(
                    start_time
                )
                else "Unknown"
            )
        )


        col4.metric(
            "End Time",
            (
                end_time.strftime(
                    "%d %b %Y %H:%M"
                )
                if pd.notna(
                    end_time
                )
                else "Active"
            )
        )


        # ----------------------------------------------------
        # SESSION RECORDS
        # ----------------------------------------------------

        if data.empty:

            session_records = (
                pd.DataFrame()
            )

        else:

            session_records = data[
                data[
                    "session_id"
                ]
                == selected_session_id
            ].copy()


        if session_records.empty:

            st.info(
                "No engagement records are "
                "available for this session."
            )

        else:

            st.divider()

            st.subheader(
                "Filters"
            )


            valid_times = (
                session_records[
                    "timestamp"
                ]
                .dropna()
            )


            filtered = (
                session_records.copy()
            )


            # ------------------------------------------------
            # DATE FILTER
            # ------------------------------------------------

            if not valid_times.empty:

                min_date = (
                    valid_times
                    .min()
                    .date()
                )

                max_date = (
                    valid_times
                    .max()
                    .date()
                )


                selected_date = (
                    st.date_input(
                        "Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                )


                filtered = filtered[
                    filtered[
                        "timestamp"
                    ]
                    .dt
                    .date
                    == selected_date
                ]


            # ------------------------------------------------
            # TIME FILTER
            # ------------------------------------------------

            time_col1, time_col2 = (
                st.columns(2)
            )


            with time_col1:

                start_filter = (
                    st.time_input(
                        "From Time",
                        value=(
                            valid_times
                            .min()
                            .time()
                            if not valid_times.empty
                            else pd.Timestamp(
                                "00:00"
                            ).time()
                        )
                    )
                )


            with time_col2:

                end_filter = (
                    st.time_input(
                        "To Time",
                        value=(
                            valid_times
                            .max()
                            .time()
                            if not valid_times.empty
                            else pd.Timestamp(
                                "23:59"
                            ).time()
                        )
                    )
                )


            if not filtered.empty:

                filtered = filtered[
                    filtered[
                        "timestamp"
                    ]
                    .dt
                    .time
                    .between(
                        start_filter,
                        end_filter
                    )
                ]


            # ------------------------------------------------
            # STUDENT FILTER
            # ------------------------------------------------

            recognised_students = (
                session_records[
                    session_records[
                        "registered_student_id"
                    ].notna()
                ][
                    [
                        "registered_student_id",
                        "student_name",
                        "student_number"
                    ]
                ]
                .drop_duplicates()
            )


            student_options = {
                "All Students": None
            }


            for _, student in (
                recognised_students
                .iterrows()
            ):

                label = (
                    f"{student['student_name']} - "
                    f"{student['student_number']}"
                )

                student_options[
                    label
                ] = student[
                    "registered_student_id"
                ]


            selected_student_label = (
                st.selectbox(
                    "Student",
                    list(
                        student_options.keys()
                    )
                )
            )


            selected_student_id = (
                student_options[
                    selected_student_label
                ]
            )


            if (
                selected_student_id
                is not None
            ):

                filtered = filtered[
                    filtered[
                        "registered_student_id"
                    ]
                    == selected_student_id
                ]


            # ------------------------------------------------
            # FILTER RESULT
            # ------------------------------------------------

            st.divider()


            if filtered.empty:

                st.info(
                    "No engagement records "
                    "available for this selection."
                )

            else:

                registered_count = (
                    filtered[
                        "registered_student_id"
                    ]
                    .dropna()
                    .nunique()
                )


                record_count = len(
                    filtered
                )


                average_score = (
                    filtered[
                        "engagement_score"
                    ]
                    .mean()
                )


                low_count = len(
                    filtered[
                        filtered[
                            "engagement_score"
                        ]
                        < LOW_THRESHOLD
                    ]
                )


                col1, col2, col3, col4 = (
                    st.columns(4)
                )


                col1.metric(
                    "Registered Students",
                    registered_count
                )

                col2.metric(
                    "Engagement Records",
                    record_count
                )

                col3.metric(
                    "Average Engagement",
                    f"{average_score:.1f}%"
                )

                col4.metric(
                    "Low Engagement",
                    low_count
                )


                # ------------------------------------------------
                # SESSION ENGAGEMENT TREND
                # ------------------------------------------------

                st.subheader(
                    "Session Engagement Trend"
                )


                trend = (
                    filtered
                    .dropna(
                        subset=[
                            "timestamp",
                            "engagement_score"
                        ]
                    )
                    .sort_values(
                        "timestamp"
                    )
                    .set_index(
                        "timestamp"
                    )
                )


                if not trend.empty:

                    st.line_chart(
                        trend[
                            "engagement_score"
                        ],
                        use_container_width=True
                    )


                # ------------------------------------------------
                # SESSION RECORDS
                # ------------------------------------------------

                st.subheader(
                    "Session Records"
                )


                session_display = (
                    filtered[
                        [
                            "timestamp",
                            "student_name",
                            "student_number",
                            "engagement_score",
                            "status",
                            "orientation",
                            "track_id"
                        ]
                    ]
                    .copy()
                )


                session_display.columns = [
                    "Time",
                    "Student Name",
                    "Student Number",
                    "Score",
                    "Status",
                    "Orientation",
                    "Track ID"
                ]


                st.dataframe(
                    session_display
                    .sort_values(
                        "Time",
                        ascending=False
                    ),
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