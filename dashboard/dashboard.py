import os
import sys
import sqlite3
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database

DB_PATH = os.path.join(PROJECT_ROOT, "data", "cems.db")

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
    engagement_records.student_id = temporary tracker ID
    engagement_records.registered_student_id = real student ID
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

        data = pd.read_sql_query(query, conn)
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

        # Ignore old incomplete/collecting states for analytics
        data = data[
            ~data["status"]
            .astype(str)
            .str.lower()
            .str.contains("collecting", na=False)
        ].copy()

        return data

    except Exception as e:
        st.error(f"Unable to load engagement data: {e}")
        return pd.DataFrame()


def get_active_session_safe():
    try:
        return database.get_active_session_id()
    except Exception:
        return None


# ============================================================
# DISPLAY HELPERS
# ============================================================

def student_label(row):
    name = row.get("student_name", "Unknown")
    number = row.get("student_number", "Unknown")

    return f"{name} - {number}"


def session_label(row):
    session_id = row.get("session_id", "")

    unit = row.get("unit_code", "")
    classroom = row.get("classroom_name", "")
    start = row.get("start_time")

    if pd.notna(start):
        date_text = start.strftime("%d %b %Y")
        time_text = start.strftime("%H:%M")
    else:
        date_text = "Unknown date"
        time_text = ""

    return (
        f"{unit} | {classroom} | "
        f"{date_text} | {time_text} | Session {session_id}"
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
    st.sidebar.info("No active session")


# ============================================================
# HEADER
# ============================================================

st.title("📊 CEMS Dashboard")
st.caption("Classroom Engagement Monitoring System")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Dashboard Overview")

    # Registered students means students table,
    # NOT temporary tracker IDs.
    total_registered = len(students_df)

    if data.empty:

        col1, col2, col3, col4, col5 = st.columns(5)

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
            active_session if active_session else "None"
        )

        st.info(
            "No engagement records are currently available."
        )

    else:

        average_engagement = (
            data["engagement_score"].mean()
        )

        engaged_count = len(
            data[
                data["status"]
                .astype(str)
                .str.lower()
                .eq("engaged")
            ]
        )

        low_count = len(
            data[
                data["engagement_score"] < LOW_THRESHOLD
            ]
        )

        if active_session is not None:
            latest_session = active_session

        elif data["session_id"].notna().any():
            latest_session = int(
                data["session_id"].dropna().max()
            )

        else:
            latest_session = "None"

        col1, col2, col3, col4, col5 = st.columns(5)

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

        st.subheader("📈 Engagement Trend")

        trend = (
            data.dropna(
                subset=[
                    "timestamp",
                    "engagement_score"
                ]
            )
            .sort_values("timestamp")
        )

        if trend.empty:

            st.info(
                "No valid engagement trend data available."
            )

        else:

            trend_chart = (
                trend[
                    [
                        "timestamp",
                        "engagement_score"
                    ]
                ]
                .set_index("timestamp")
            )

            st.line_chart(
                trend_chart,
                use_container_width=True
            )

        # ----------------------------------------------------
        # RECENT RECORDS
        # ----------------------------------------------------

        st.subheader("🧑‍🎓 Recent Engagement Records")

        recent = (
            data.sort_values(
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

        st.subheader("🚨 Students Requiring Attention")

        low_records = data[
            data["engagement_score"] < LOW_THRESHOLD
        ].copy()

        if low_records.empty:

            st.success(
                "No low-engagement students detected."
            )

        else:

            alert_summary = (
                low_records[
                    low_records["registered_student_id"]
                    .notna()
                ]
                .groupby(
                    [
                        "registered_student_id",
                        "student_name",
                        "student_number"
                    ],
                    dropna=False
                )
                .agg(
                    Low_Records=(
                        "engagement_score",
                        "count"
                    ),
                    Average_Score=(
                        "engagement_score",
                        "mean"
                    ),
                    Latest_Low=(
                        "timestamp",
                        "max"
                    )
                )
                .reset_index()
            )

            alert_summary[
                "Average_Score"
            ] = alert_summary[
                "Average_Score"
            ].round(1)

            for _, row in alert_summary.iterrows():

                if row["Low_Records"] >= 2:
                    message = (
                        f"⚠️ {row['student_name']} "
                        f"({row['student_number']}) — "
                        f"sustained low engagement "
                        f"({row['Low_Records']} low records)"
                    )
                else:
                    message = (
                        f"⚠️ {row['student_name']} "
                        f"({row['student_number']}) — "
                        f"low engagement detected"
                    )

                st.warning(message)


# ============================================================
# LIVE CLASSROOM
# ============================================================

elif page == "Live Classroom":

    st.header("Live Classroom")

    if active_session is None:

        st.warning(
            "There is currently no active classroom session."
        )

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
                    "No engagement records are available "
                    "for the active session."
                )

            else:

                # Latest record for each REAL registered student.
                recognised = session_data[
                    session_data[
                        "registered_student_id"
                    ].notna()
                ].copy()

                latest_students = (
                    recognised
                    .sort_values("timestamp")
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
                        latest_students["status"]
                        .astype(str)
                        .str.lower()
                        .eq("engaged")
                    ]
                )

                neutral = len(
                    latest_students[
                        latest_students["status"]
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

                col1, col2, col3, col4 = st.columns(4)

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

                st.subheader(
                    "Recognised Students"
                )

                if latest_students.empty:

                    st.info(
                        "No recognised registered students "
                        "are currently available."
                    )

                else:

                    live_display = latest_students[
                        [
                            "student_name",
                            "student_number",
                            "engagement_score",
                            "status",
                            "orientation",
                            "timestamp"
                        ]
                    ].copy()

                    live_display[
                        "timestamp"
                    ] = live_display[
                        "timestamp"
                    ].dt.strftime("%H:%M:%S")

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
                    .sort_values("timestamp")
                    .set_index("timestamp")
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

    st.header("Student History")

    if students_df.empty:

        st.warning(
            "No registered students are available."
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
        ] = selectable_students.apply(
            student_label,
            axis=1
        )

        labels = selectable_students[
            "label"
        ].tolist()

        selected_label = st.selectbox(
            "Select Student",
            labels
        )

        selected_row = selectable_students[
            selectable_students["label"]
            == selected_label
        ].iloc[0]

        selected_id = selected_row[
            "registered_student_id"
        ]

        student_data = data[
            data["registered_student_id"]
            == selected_id
        ].copy() if not data.empty else pd.DataFrame()

        st.caption(
            f"Registered Student ID: {selected_id}"
        )

        if student_data.empty:

            st.info(
                "No engagement records are available "
                "for this student."
            )

        else:

            average_score = (
                student_data[
                    "engagement_score"
                ].mean()
            )

            record_count = len(student_data)

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

            col1, col2, col3, col4 = st.columns(4)

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

            # ----------------------------------------------
            # ENGAGEMENT TREND
            # ----------------------------------------------

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
                .sort_values("timestamp")
                .set_index("timestamp")
            )

            if not student_chart.empty:

                st.line_chart(
                    student_chart[
                        "engagement_score"
                    ],
                    use_container_width=True
                )

            # ----------------------------------------------
            # STATUS HISTORY
            # ----------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Status History"
                )

                status_counts = (
                    student_data[
                        "status"
                    ]
                    .fillna("Unknown")
                    .value_counts()
                )

                st.bar_chart(
                    status_counts
                )

            with col2:

                st.subheader(
                    "Orientation History"
                )

                orientation_counts = (
                    student_data[
                        "orientation"
                    ]
                    .fillna("Unknown")
                    .value_counts()
                )

                st.bar_chart(
                    orientation_counts
                )

            # ----------------------------------------------
            # SESSION HISTORY
            # ----------------------------------------------

            st.subheader(
                "Session History"
            )

            session_summary = (
                student_data
                .groupby(
                    "session_id",
                    dropna=False
                )["engagement_score"]
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
            ] = session_summary[
                "Average"
            ].round(1)

            st.dataframe(
                session_summary,
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "All Student Records"
            )

            student_display = student_data[
                [
                    "timestamp",
                    "engagement_score",
                    "status",
                    "orientation",
                    "session_id",
                    "track_id"
                ]
            ].copy()

            student_display.columns = [
                "Time",
                "Score",
                "Status",
                "Orientation",
                "Session",
                "Track ID"
            ]

            st.dataframe(
                student_display.sort_values(
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

    st.header("Weekly Analytics")

    if data.empty:

        st.warning(
            "No engagement data is available."
        )

    else:

        analytics_data = data.dropna(
            subset=[
                "timestamp",
                "engagement_score"
            ]
        ).copy()

        if analytics_data.empty:

            st.info(
                "No valid analytics data is available."
            )

        else:

            # ----------------------------------------------
            # OVERALL TREND
            # ----------------------------------------------

            st.subheader(
                "Average Engagement Over Time"
            )

            trend = (
                analytics_data
                .sort_values("timestamp")
                .set_index("timestamp")
            )

            st.line_chart(
                trend[
                    "engagement_score"
                ],
                use_container_width=True
            )

            # ----------------------------------------------
            # STATUS DISTRIBUTION
            # ----------------------------------------------

            st.subheader(
                "Engagement Status Distribution"
            )

            status_distribution = (
                analytics_data[
                    "status"
                ]
                .fillna("Unknown")
                .value_counts()
            )

            st.bar_chart(
                status_distribution
            )

            # ----------------------------------------------
            # STUDENT COMPARISON
            # ----------------------------------------------

            st.subheader(
                "Student Average Comparison"
            )

            recognised = analytics_data[
                analytics_data[
                    "registered_student_id"
                ].notna()
            ].copy()

            if recognised.empty:

                st.info(
                    "No recognised student data available."
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
                    )["engagement_score"]
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
                ] = student_average[
                    "engagement_score"
                ].round(1)

                st.bar_chart(
                    student_average.set_index(
                        "Student"
                    )["Average"]
                )

            # ----------------------------------------------
            # DAILY AVERAGE
            # ----------------------------------------------

            st.subheader(
                "Daily Average"
            )

            analytics_data[
                "date"
            ] = analytics_data[
                "timestamp"
            ].dt.date

            daily = (
                analytics_data
                .groupby("date")[
                    "engagement_score"
                ]
                .mean()
                .reset_index()
            )

            daily[
                "Average"
            ] = daily[
                "engagement_score"
            ].round(1)

            st.line_chart(
                daily.set_index(
                    "date"
                )["Average"]
            )

            # ----------------------------------------------
            # WEEKLY AVERAGE
            # ----------------------------------------------

            st.subheader(
                "Weekly Average"
            )

            analytics_data[
                "week"
            ] = (
                analytics_data[
                    "timestamp"
                ]
                .dt.to_period("W")
                .astype(str)
            )

            weekly = (
                analytics_data
                .groupby("week")[
                    "engagement_score"
                ]
                .agg(
                    Average="mean",
                    Records="count"
                )
                .reset_index()
            )

            weekly[
                "Average"
            ] = weekly[
                "Average"
            ].round(1)

            st.line_chart(
                weekly.set_index(
                    "week"
                )["Average"]
            )

            st.dataframe(
                weekly,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# SESSION HISTORY
# ============================================================

elif page == "Session History":

    st.header("Session History")

    if sessions_df.empty:

        st.warning(
            "No sessions have been created."
        )

    else:

        session_options = {}

        for _, row in sessions_df.iterrows():

            label = session_label(row)

            session_options[
                label
            ] = row["session_id"]

        selected_session_label = st.selectbox(
            "Select Session",
            list(session_options.keys())
        )

        selected_session_id = (
            session_options[
                selected_session_label
            ]
        )

        selected_session = sessions_df[
            sessions_df["session_id"]
            == selected_session_id
        ].iloc[0]

        # ----------------------------------------------
        # SESSION INFORMATION
        # ----------------------------------------------

        st.subheader(
            "Session Information"
        )

        col1, col2, col3, col4 = st.columns(4)

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

        start_time = selected_session[
            "start_time"
        ]

        end_time = selected_session[
            "end_time"
        ]

        col3.metric(
            "Start Time",
            (
                start_time.strftime(
                    "%d %b %Y %H:%M"
                )
                if pd.notna(start_time)
                else "Unknown"
            )
        )

        col4.metric(
            "End Time",
            (
                end_time.strftime(
                    "%d %b %Y %H:%M"
                )
                if pd.notna(end_time)
                else "Active"
            )
        )

        # ----------------------------------------------
        # SESSION RECORDS
        # ----------------------------------------------

        if data.empty:

            session_records = pd.DataFrame()

        else:

            session_records = data[
                data["session_id"]
                == selected_session_id
            ].copy()

        if session_records.empty:

            st.info(
                "No engagement records are available "
                "for this session."
            )

        else:

            st.divider()

            st.subheader(
                "Filters"
            )

            valid_times = session_records[
                "timestamp"
            ].dropna()

            filtered = session_records.copy()

            # ------------------------------------------
            # DATE FILTER
            # ------------------------------------------

            if not valid_times.empty:

                min_date = valid_times.min().date()
                max_date = valid_times.max().date()

                selected_date = st.date_input(
                    "Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )

                filtered = filtered[
                    filtered[
                        "timestamp"
                    ].dt.date
                    == selected_date
                ]

            # ------------------------------------------
            # TIME FILTER
            # ------------------------------------------

            time_col1, time_col2 = st.columns(2)

            with time_col1:

                start_filter = st.time_input(
                    "From Time",
                    value=(
                        valid_times.min().time()
                        if not valid_times.empty
                        else pd.Timestamp(
                            "00:00"
                        ).time()
                    )
                )

            with time_col2:

                end_filter = st.time_input(
                    "To Time",
                    value=(
                        valid_times.max().time()
                        if not valid_times.empty
                        else pd.Timestamp(
                            "23:59"
                        ).time()
                    )
                )

            if not filtered.empty:

                filtered = filtered[
                    filtered[
                        "timestamp"
                    ].dt.time
                    .between(
                        start_filter,
                        end_filter
                    )
                ]

            # ------------------------------------------
            # STUDENT FILTER
            # ------------------------------------------

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
                recognised_students.iterrows()
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

            selected_student_label = st.selectbox(
                "Student",
                list(student_options.keys())
            )

            selected_student_id = (
                student_options[
                    selected_student_label
                ]
            )

            if selected_student_id is not None:

                filtered = filtered[
                    filtered[
                        "registered_student_id"
                    ]
                    == selected_student_id
                ]

            # ------------------------------------------
            # FILTER RESULT
            # ------------------------------------------

            st.divider()

            if filtered.empty:

                st.info(
                    "No engagement records available "
                    "for this selection."
                )

            else:

                registered_count = (
                    filtered[
                        "registered_student_id"
                    ]
                    .dropna()
                    .nunique()
                )

                record_count = len(filtered)

                average_score = (
                    filtered[
                        "engagement_score"
                    ].mean()
                )

                low_count = len(
                    filtered[
                        filtered[
                            "engagement_score"
                        ] < LOW_THRESHOLD
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
                    .sort_values("timestamp")
                    .set_index("timestamp")
                )

                if not trend.empty:

                    st.line_chart(
                        trend[
                            "engagement_score"
                        ],
                        use_container_width=True
                    )

                st.subheader(
                    "Session Records"
                )

                session_display = filtered[
                    [
                        "timestamp",
                        "student_name",
                        "student_number",
                        "engagement_score",
                        "status",
                        "orientation",
                        "track_id"
                    ]
                ].copy()

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
                    session_display.sort_values(
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