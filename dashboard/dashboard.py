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
    st.info(
        "Live Classroom will be implemented next."
    )


elif page == "Student History":

    st.title("👨‍🎓 Student History")
    st.info(
        "Student History will be implemented next."
    )


elif page == "Weekly Analytics":

    st.title("📈 Weekly Analytics")
    st.info(
        "Weekly Analytics will be implemented next."
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