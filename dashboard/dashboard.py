import os
import pandas as pd
import streamlit as st


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Classroom Engagement Monitoring System",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("📊 Classroom Engagement Monitoring System")

st.subheader("Classroom Overview")


# ---------------------------------------------------
# Load Engagement Data
# ---------------------------------------------------

csv_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "engagement.csv"
)


if not os.path.exists(csv_path):

    st.error(
        "engagement.csv was not found. "
        "Please make sure it is inside the data folder."
    )

    st.stop()


data = pd.read_csv(csv_path)


# ---------------------------------------------------
# Check Required Columns
# ---------------------------------------------------

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
        f"Missing columns in engagement.csv: {missing_columns}"
    )

    st.stop()


# ---------------------------------------------------
# Calculate Statistics
# ---------------------------------------------------

average_engagement = data["engagement_score"].mean()

students_detected = data["student_id"].nunique()

low_engagement = data[
    data["engagement_score"] < 60
]


# ---------------------------------------------------
# Dashboard Metrics
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        label="Average Engagement",
        value=f"{average_engagement:.0f}%"
    )


with col2:

    st.metric(
        label="Students Detected",
        value=students_detected
    )


with col3:

    st.metric(
        label="Low Engagement Records",
        value=len(low_engagement)
    )


# ---------------------------------------------------
# Engagement Trend
# ---------------------------------------------------

st.subheader("📈 Engagement Trend")


chart_data = data[
    ["engagement_score"]
]


st.line_chart(
    chart_data
)


# ---------------------------------------------------
# Student Engagement Table
# ---------------------------------------------------

st.subheader("👨‍🎓 Student Engagement")


display_data = data[
    [
        "student_id",
        "engagement_score",
        "status",
        "timestamp"
    ]
]


st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------
# Alerts
# ---------------------------------------------------

st.subheader("🚨 Low Engagement Alerts")


if low_engagement.empty:

    st.success(
        "No low engagement students detected."
    )

else:

    for _, row in low_engagement.iterrows():

        st.warning(
            f"Student {row['student_id']} "
            f"has {row['engagement_score']}% "
            f"engagement – {row['status']}"
        )