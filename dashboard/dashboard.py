import os
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CEMS Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    """
    Load engagement data from the existing CSV file.

    This is temporary for development.
    Later this function can be replaced with SQLite queries.
    """

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "engagement.csv"
    )

    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        data = pd.read_csv(csv_path)

        required_columns = [
            "student_id",
            "engagement_score",
            "status",
            "timestamp"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing columns in engagement.csv: "
                f"{missing_columns}"
            )
            return pd.DataFrame()

        # Make sure engagement scores are numeric
        data["engagement_score"] = pd.to_numeric(
            data["engagement_score"],
            errors="coerce"
        )

        # Remove invalid records
        data = data.dropna(
            subset=["engagement_score"]
        )

        return data

    except Exception as error:
        st.error(
            f"Unable to load engagement data: {error}"
        )
        return pd.DataFrame()


data = load_data()


# =========================================================
# NAVIGATION
# =========================================================

st.sidebar.title("📊 CEMS")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live Classroom",
        "Student History",
        "Weekly Analytics",
        "Classroom Comparison",
        "Session History"
    ]
)


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.title("📊 CEMS Dashboard")
    st.caption(
        "Classroom Engagement Monitoring System"
    )

    st.header("Dashboard Overview")

    # -----------------------------------------------------
    # EMPTY DATA HANDLING
    # -----------------------------------------------------

    if data.empty:

        st.warning(
            "No engagement data is currently available."
        )

        st.info(
            "The dashboard is ready, but there are no "
            "engagement records to display yet."
        )

        st.stop()

    # -----------------------------------------------------
    # CALCULATE METRICS
    # -----------------------------------------------------

    average_engagement = data[
        "engagement_score"
    ].mean()

    students_detected = data[
        "student_id"
    ].nunique()

    low_engagement = data[
        data["engagement_score"] < 60
    ]

    low_engagement_students = low_engagement[
        "student_id"
    ].nunique()

    # -----------------------------------------------------
    # CURRENT SESSION INFORMATION
    # -----------------------------------------------------

    st.subheader("Current Session")

    session_col1, session_col2, session_col3 = st.columns(3)

    with session_col1:
        st.metric(
            "Unit",
            "Not available yet"
        )

    with session_col2:
        st.metric(
            "Classroom",
            "Not available yet"
        )

    with session_col3:
        st.metric(
            "Session Status",
            "Development"
        )

    st.divider()

    # -----------------------------------------------------
    # MAIN METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Engagement",
            f"{average_engagement:.0f}%"
        )

    with col2:
        st.metric(
            "Students",
            students_detected
        )

    with col3:
        st.metric(
            "Low Engagement Students",
            low_engagement_students
        )

    # -----------------------------------------------------
    # ENGAGEMENT TREND
    # -----------------------------------------------------

    st.subheader("📈 Engagement Trend")

    chart_data = data[
        ["timestamp", "engagement_score"]
    ].copy()

    chart_data["timestamp"] = pd.to_datetime(
        chart_data["timestamp"],
        errors="coerce"
    )

    chart_data = chart_data.dropna(
        subset=["timestamp"]
    )

    chart_data = chart_data.sort_values(
        "timestamp"
    )

    if chart_data.empty:

        st.info(
            "There is not enough timestamp data "
            "to display the engagement trend."
        )

    else:

        chart_data = chart_data.set_index(
            "timestamp"
        )

        st.line_chart(
            chart_data["engagement_score"],
            width="stretch"
        )

    # -----------------------------------------------------
    # STUDENT SUMMARY
    # -----------------------------------------------------

    st.subheader("👨‍🎓 Student Engagement")

    student_summary = (
        data
        .groupby("student_id")
        .agg(
            Average_Engagement=(
                "engagement_score",
                "mean"
            ),
            Records=(
                "engagement_score",
                "count"
            )
        )
        .reset_index()
    )

    student_summary[
        "Average_Engagement"
    ] = student_summary[
        "Average_Engagement"
    ].round(1)

    student_summary[
        "Status"
    ] = student_summary[
        "Average_Engagement"
    ].apply(
        lambda score:
            "Low"
            if score < 60
            else "Normal"
    )

    st.dataframe(
        student_summary,
        width="stretch",
        hide_index=True
    )

    # -----------------------------------------------------
    # LOW ENGAGEMENT ALERTS
    # -----------------------------------------------------

    st.subheader("🚨 Low Engagement Alerts")

    if low_engagement.empty:

        st.success(
            "No low engagement students detected."
        )

    else:

        for student_id in (
            low_engagement["student_id"]
            .unique()
        ):

            student_records = low_engagement[
                low_engagement["student_id"]
                == student_id
            ]

            average_score = student_records[
                "engagement_score"
            ].mean()

            st.warning(
                f"Student {student_id} — "
                f"Average low engagement: "
                f"{average_score:.0f}%"
            )


# =========================================================
# PLACEHOLDER PAGES
# =========================================================

elif page == "Live Classroom":

    st.title("🎥 Live Classroom")

    st.caption(
        "Latest available engagement activity for tracked students"
    )

    if data.empty:

        st.warning(
            "No engagement records are currently available."
        )

        st.info(
            "Student tracking data will appear here "
            "when engagement records are received."
        )

    else:

        # Get the latest record for each student
        live_data = data.copy()

        live_data["timestamp"] = pd.to_datetime(
            live_data["timestamp"],
            errors="coerce"
        )

        live_data = live_data.dropna(
            subset=["timestamp"]
        )

        live_data = live_data.sort_values(
            "timestamp"
        )

        latest_students = (
            live_data
            .groupby("student_id")
            .tail(1)
            .reset_index(drop=True)
        )

        # ---------------------------------------------
        # LIVE METRICS
        # ---------------------------------------------

        total_students = latest_students[
            "student_id"
        ].nunique()

        average_engagement = latest_students[
            "engagement_score"
        ].mean()

        low_students = latest_students[
            latest_students["engagement_score"] < 60
        ]

        low_count = low_students[
            "student_id"
        ].nunique()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Students Currently Tracked",
                total_students
            )

        with col2:
            st.metric(
                "Current Average Engagement",
                f"{average_engagement:.0f}%"
            )

        with col3:
            st.metric(
                "Low Engagement Students",
                low_count
            )

        st.divider()

        # ---------------------------------------------
        # CURRENT STUDENT ACTIVITY
        # ---------------------------------------------

        st.subheader("👨‍🎓 Current Student Activity")

        display_data = latest_students[
            [
                "student_id",
                "engagement_score",
                "status",
                "timestamp"
            ]
        ].copy()

        display_data = display_data.rename(
            columns={
                "student_id": "Student",
                "engagement_score": "Engagement",
                "status": "Status",
                "timestamp": "Last Update"
            }
        )

        display_data["Engagement"] = (
            display_data["Engagement"]
            .round(0)
            .astype(int)
            .astype(str)
            + "%"
        )

        display_data["Last Update"] = (
            display_data["Last Update"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True
        )

        # ---------------------------------------------
        # LOW ENGAGEMENT ALERTS
        # ---------------------------------------------

        st.subheader("🚨 Low Engagement Alerts")

        if low_students.empty:

            st.success(
                "No students are currently below "
                "the 60% engagement threshold."
            )

        else:

            for _, row in low_students.iterrows():

                st.warning(
                    f"Student {row['student_id']} — "
                    f"{row['engagement_score']:.0f}% engagement "
                    f"({row['status']})"
                )

        # ---------------------------------------------
        # LAST UPDATE
        # ---------------------------------------------

        latest_timestamp = live_data[
            "timestamp"
        ].max()

        st.caption(
            f"Latest data received: "
            f"{latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )

elif page == "Student History":

    st.title("👨‍🎓 Student History")

    st.caption(
        "Individual student engagement history and session activity"
    )

    # -----------------------------------------------------
    # CHECK DATA
    # -----------------------------------------------------

    if data.empty:

        st.warning(
            "No student engagement data is currently available."
        )

    else:

        # -------------------------------------------------
        # STUDENT SELECTION
        # -------------------------------------------------

        student_ids = sorted(
            data["student_id"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_student = st.selectbox(
            "Select Student",
            student_ids
        )

        # -------------------------------------------------
        # FILTER SELECTED STUDENT
        # -------------------------------------------------

        student_data = data[
            data["student_id"] == selected_student
        ].copy()

        student_data["timestamp"] = pd.to_datetime(
            student_data["timestamp"],
            errors="coerce"
        )

        student_data = student_data.dropna(
            subset=["timestamp"]
        )

        student_data = student_data.sort_values(
            "timestamp"
        )

        # -------------------------------------------------
        # STUDENT METRICS
        # -------------------------------------------------

        average_engagement = student_data[
            "engagement_score"
        ].mean()

        total_records = len(student_data)

        low_records = student_data[
            student_data["engagement_score"] < 60
        ]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Student",
                str(selected_student)
            )

        with col2:

            st.metric(
                "Average Engagement",
                f"{average_engagement:.0f}%"
            )

        with col3:

            st.metric(
                "Low Engagement Periods",
                len(low_records)
            )

        st.divider()

        # -------------------------------------------------
        # ENGAGEMENT HISTORY CHART
        # -------------------------------------------------

        st.subheader("📈 Engagement Over Time")

        chart_data = student_data[
            [
                "timestamp",
                "engagement_score"
            ]
        ].copy()

        chart_data = chart_data.set_index(
            "timestamp"
        )

        st.line_chart(
            chart_data["engagement_score"],
            width="stretch"
        )

        # -------------------------------------------------
        # ENGAGEMENT STATUS
        # -------------------------------------------------

        st.subheader("📊 Engagement Records")

        display_data = student_data[
            [
                "timestamp",
                "engagement_score",
                "status"
            ]
        ].copy()

        display_data = display_data.rename(
            columns={
                "timestamp": "Time",
                "engagement_score": "Engagement",
                "status": "Status"
            }
        )

        display_data["Engagement"] = (
            display_data["Engagement"]
            .round(0)
            .astype(int)
            .astype(str)
            + "%"
        )

        display_data["Time"] = (
            pd.to_datetime(
                display_data["Time"]
            ).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True
        )

        # -------------------------------------------------
        # LOW ENGAGEMENT PERIODS
        # -------------------------------------------------

        st.subheader("🚨 Low Engagement Periods")

        if low_records.empty:

            st.success(
                "No low-engagement periods recorded "
                "for this student."
            )

        else:

            for _, row in low_records.iterrows():

                st.warning(
                    f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} — "
                    f"{row['engagement_score']:.0f}% engagement "
                    f"({row['status']})"
                )

elif page == "Weekly Analytics":

    st.title("📈 Weekly Analytics")

    st.caption(
        "Weekly classroom engagement trends based on stored records"
    )

    # -----------------------------------------------------
    # CHECK DATA
    # -----------------------------------------------------

    if data.empty:

        st.warning(
            "No engagement data is currently available."
        )

    else:

        weekly_data = data.copy()

        # Convert timestamp to datetime
        weekly_data["timestamp"] = pd.to_datetime(
            weekly_data["timestamp"],
            errors="coerce"
        )

        weekly_data = weekly_data.dropna(
            subset=["timestamp"]
        )

        if weekly_data.empty:

            st.warning(
                "No valid timestamp data is available "
                "for weekly analytics."
            )

        else:

            # -------------------------------------------------
            # LAST 7 DAYS
            # -------------------------------------------------

            latest_date = weekly_data[
                "timestamp"
            ].max()

            start_date = latest_date - pd.Timedelta(
                days=6
            )

            weekly_data = weekly_data[
                weekly_data["timestamp"] >= start_date
            ].copy()

            # -------------------------------------------------
            # WEEKLY METRICS
            # -------------------------------------------------

            weekly_average = weekly_data[
                "engagement_score"
            ].mean()

            total_records = len(
                weekly_data
            )

            low_records = weekly_data[
                weekly_data["engagement_score"] < 60
            ]

            low_students = low_records[
                "student_id"
            ].nunique()

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Weekly Average",
                    f"{weekly_average:.0f}%"
                )

            with col2:

                st.metric(
                    "Engagement Records",
                    total_records
                )

            with col3:

                st.metric(
                    "Low Engagement Students",
                    low_students
                )

            st.divider()

            # -------------------------------------------------
            # DAILY AVERAGES
            # -------------------------------------------------

            weekly_data["Date"] = (
                weekly_data["timestamp"]
                .dt.date
            )

            daily_average = (
                weekly_data
                .groupby("Date")[
                    "engagement_score"
                ]
                .mean()
                .reset_index()
            )

            daily_average[
                "Average Engagement"
            ] = daily_average[
                "engagement_score"
            ].round(1)

            # -------------------------------------------------
            # WEEKLY ENGAGEMENT GRAPH
            # -------------------------------------------------

            st.subheader(
                "📊 Engagement Trend"
            )

            chart_data = (
                daily_average[
                    [
                        "Date",
                        "Average Engagement"
                    ]
                ]
                .set_index("Date")
            )

            st.line_chart(
                chart_data,
                width="stretch"
            )

            # -------------------------------------------------
            # HIGHEST / LOWEST DAY
            # -------------------------------------------------

            if not daily_average.empty:

                highest_row = daily_average.loc[
                    daily_average[
                        "Average Engagement"
                    ].idxmax()
                ]

                lowest_row = daily_average.loc[
                    daily_average[
                        "Average Engagement"
                    ].idxmin()
                ]

                col1, col2 = st.columns(2)

                with col1:

                    st.success(
                        f"Highest Engagement: "
                        f"{highest_row['Date']} — "
                        f"{highest_row['Average Engagement']:.0f}%"
                    )

                with col2:

                    st.warning(
                        f"Lowest Engagement: "
                        f"{lowest_row['Date']} — "
                        f"{lowest_row['Average Engagement']:.0f}%"
                    )

            # -------------------------------------------------
            # DAILY ANALYTICS TABLE
            # -------------------------------------------------

            st.subheader(
                "📅 Daily Engagement Summary"
            )

            display_daily = daily_average[
                [
                    "Date",
                    "Average Engagement"
                ]
            ].copy()

            display_daily[
                "Average Engagement"
            ] = (
                display_daily[
                    "Average Engagement"
                ]
                .astype(str)
                + "%"
            )

            st.dataframe(
                display_daily,
                width="stretch",
                hide_index=True
            )

elif page == "Classroom Comparison":

    st.title("🏫 Classroom Comparison")
    st.info(
        "Classroom Comparison will be implemented next."
    )


elif page == "Session History":

    st.title("🕒 Session History")
    st.info(
        "Session History will be implemented next."
    )