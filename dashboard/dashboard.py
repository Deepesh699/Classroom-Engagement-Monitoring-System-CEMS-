import os
import sqlite3

import pandas as pd
import streamlit as st


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="CEMS Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CEMS Engagement Dashboard")

st.write(
    "Filter classroom engagement data by session, date, time and student."
)


# ---------------------------------------------------
# DATABASE
# ---------------------------------------------------

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "cems.db"
)

if not os.path.exists(DB_PATH):
    st.error("CEMS database was not found.")
    st.stop()


try:
    connection = sqlite3.connect(DB_PATH)

    data = pd.read_sql_query(
        """
        SELECT
            id,
            timestamp,
            student_id,
            registered_student_id,
            session_id,
            engagement_score,
            status,
            orientation
        FROM engagement_records
        ORDER BY id ASC
        """,
        connection
    )

    students = pd.read_sql_query(
        """
        SELECT *
        FROM students
        """,
        connection
    )

    connection.close()

except Exception as error:
    st.error(f"Could not load database: {error}")
    st.stop()


# ---------------------------------------------------
# EMPTY DATA
# ---------------------------------------------------

if data.empty:
    st.warning(
        "The database currently has no engagement records."
    )
    st.stop()


# ---------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------

data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    errors="coerce"
)

data = data.dropna(
    subset=["timestamp"]
)

if data.empty:
    st.warning(
        "No valid timestamped engagement records are available."
    )
    st.stop()

data["date"] = data["timestamp"].dt.date
data["time"] = data["timestamp"].dt.time


# ---------------------------------------------------
# STUDENT NAMES
# ---------------------------------------------------

student_names = {}

if not students.empty:

    for _, student in students.iterrows():

        student_id = student.iloc[0]

        if len(student) >= 3:

            student_name = student.iloc[2]

            student_names[int(student_id)] = student_name


# ---------------------------------------------------
# FILTERS
# ---------------------------------------------------

st.subheader("Filters")

filter1, filter2 = st.columns(2)
filter3, filter4 = st.columns(2)


# ---------------------------------------------------
# SESSION FILTER
# ---------------------------------------------------

session_values = sorted(
    data["session_id"]
    .dropna()
    .unique()
    .tolist()
)

session_options = ["All Sessions"] + session_values

with filter1:

    selected_session = st.selectbox(
        "Session",
        session_options
    )


filtered = data.copy()

if selected_session != "All Sessions":

    filtered = filtered[
        filtered["session_id"]
        == selected_session
    ]


# ---------------------------------------------------
# DATE FILTER
# ---------------------------------------------------

date_values = sorted(
    filtered["date"]
    .dropna()
    .unique()
    .tolist()
)

date_options = ["All Dates"] + date_values

with filter2:

    selected_date = st.selectbox(
        "Date",
        date_options
    )


if selected_date != "All Dates":

    filtered = filtered[
        filtered["date"]
        == selected_date
    ]


# ---------------------------------------------------
# TIME FILTER
# ---------------------------------------------------

with filter3:

    st.write("Time Range")

    if filtered.empty:

        st.info(
            "No time data available for the selected filters."
        )

    else:

        min_time = (
            filtered["timestamp"]
            .min()
            .time()
        )

        max_time = (
            filtered["timestamp"]
            .max()
            .time()
        )

        # Streamlit cannot create a range slider
        # when minimum and maximum are identical.
        if min_time == max_time:

            st.info(
                "Only one recorded time is available: "
                f"{min_time.strftime('%H:%M:%S')}"
            )

        else:

            selected_time = st.slider(
                "Select time range",
                min_value=min_time,
                max_value=max_time,
                value=(min_time, max_time)
            )

            start_time, end_time = selected_time

            filtered = filtered[
                filtered["time"].apply(
                    lambda value:
                    start_time <= value <= end_time
                )
            ]


# ---------------------------------------------------
# STUDENT FILTER
# ---------------------------------------------------

registered_ids = sorted(
    filtered["registered_student_id"]
    .dropna()
    .unique()
    .tolist()
)

student_options = {
    "All Students": None
}

for registered_id in registered_ids:

    registered_id = int(registered_id)

    student_name = student_names.get(
        registered_id,
        f"Student {registered_id}"
    )

    student_options[
        f"{student_name} (ID {registered_id})"
    ] = registered_id


with filter4:

    selected_student_label = st.selectbox(
        "Student",
        list(student_options.keys())
    )


selected_student = student_options[
    selected_student_label
]

if selected_student is not None:

    filtered = filtered[
        filtered["registered_student_id"]
        == selected_student
    ]


# ---------------------------------------------------
# FILTER RESULT
# ---------------------------------------------------

st.divider()

if filtered.empty:

    st.warning(
        "No engagement records match the selected filters."
    )

    st.stop()


# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

average_engagement = (
    filtered["engagement_score"].mean()
)

registered_students = (
    filtered["registered_student_id"]
    .dropna()
    .nunique()
)

low_engagement = filtered[
    filtered["engagement_score"] < 60
]


metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Average Engagement",
    f"{average_engagement:.1f}%"
)

metric2.metric(
    "Registered Students",
    registered_students
)

metric3.metric(
    "Engagement Records",
    len(filtered)
)

metric4.metric(
    "Low Engagement Records",
    len(low_engagement)
)


# ---------------------------------------------------
# ENGAGEMENT TREND
# ---------------------------------------------------

st.subheader("📈 Engagement Trend")

trend = (
    filtered[
        [
            "timestamp",
            "engagement_score"
        ]
    ]
    .sort_values("timestamp")
    .set_index("timestamp")
)

st.line_chart(
    trend
)


# ---------------------------------------------------
# STUDENT COMPARISON
# ---------------------------------------------------

st.subheader(
    "Student Average Engagement"
)

student_average = (
    filtered
    .dropna(
        subset=["registered_student_id"]
    )
    .groupby(
        "registered_student_id"
    )["engagement_score"]
    .mean()
)


if student_average.empty:

    st.info(
        "No registered student data available."
    )

else:

    student_average.index = [
        student_names.get(
            int(student_id),
            f"Student {int(student_id)}"
        )
        for student_id
        in student_average.index
    ]

    st.bar_chart(
        student_average
    )


# ---------------------------------------------------
# SESSION COMPARISON
# ---------------------------------------------------

st.subheader(
    "Session Comparison"
)

session_average = (
    filtered
    .dropna(
        subset=["session_id"]
    )
    .groupby(
        "session_id"
    )["engagement_score"]
    .mean()
)


if session_average.empty:

    st.info(
        "No session data available."
    )

else:

    session_average.index = [
        f"Session {int(session_id)}"
        for session_id
        in session_average.index
    ]

    st.bar_chart(
        session_average
    )


# ---------------------------------------------------
# ENGAGEMENT RECORDS
# ---------------------------------------------------

st.subheader(
    "Engagement Records"
)

display_data = filtered[
    [
        "timestamp",
        "registered_student_id",
        "session_id",
        "engagement_score",
        "status",
        "orientation"
    ]
].copy()


def get_student_name(student_id):

    if pd.isna(student_id):
        return "Unregistered"

    student_id = int(student_id)

    return student_names.get(
        student_id,
        f"Student {student_id}"
    )


display_data["Student"] = (
    display_data[
        "registered_student_id"
    ].apply(
        get_student_name
    )
)


display_data = display_data[
    [
        "timestamp",
        "Student",
        "session_id",
        "engagement_score",
        "status",
        "orientation"
    ]
]


display_data.columns = [
    "Timestamp",
    "Student",
    "Session",
    "Engagement Score",
    "Status",
    "Orientation"
]


st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------
# LOW ENGAGEMENT ALERTS
# ---------------------------------------------------

st.subheader(
    "🚨 Low Engagement Alerts"
)


if low_engagement.empty:

    st.success(
        "No low-engagement records "
        "for the selected filters."
    )

else:

    for _, row in low_engagement.iterrows():

        student_name = get_student_name(
            row["registered_student_id"]
        )

        st.warning(
            f"{student_name}: "
            f"{row['engagement_score']}% engagement "
            f"({row['status']}) at "
            f"{row['timestamp']}"
        )