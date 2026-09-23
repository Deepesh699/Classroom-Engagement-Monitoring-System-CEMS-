import os
import sqlite3
import subprocess
import sys

import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="CEMS Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Classroom Engagement Monitoring System")
st.caption("CEMS Control Centre and Engagement Dashboard")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cems.db"
)

VENV_PYTHON = os.path.join(
    PROJECT_ROOT,
    ".venv",
    "Scripts",
    "python.exe"
)


# ============================================================
# SESSION STATE
# ============================================================

if "show_analytics" not in st.session_state:
    st.session_state["show_analytics"] = False


# ============================================================
# GET PYTHON EXECUTABLE
# ============================================================

def get_python_executable():

    if os.path.exists(VENV_PYTHON):
        return VENV_PYTHON

    return sys.executable


# ============================================================
# OPEN NORMAL PYTHON / OPENCV PROGRAM
# ============================================================

def open_program(script_name):
    """
    Opens interactive Python/OpenCV programs such as
    face registration and live monitoring.
    """

    script_path = os.path.join(
        PROJECT_ROOT,
        script_name
    )

    if not os.path.exists(script_path):

        st.error(
            f"{script_name} could not be found."
        )

        return False

    try:

        python_executable = get_python_executable()

        if os.name == "nt":

            subprocess.Popen(
                [
                    "cmd.exe",
                    "/k",
                    python_executable,
                    script_path
                ],
                cwd=PROJECT_ROOT,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

        else:

            subprocess.Popen(
                [
                    python_executable,
                    script_path
                ],
                cwd=PROJECT_ROOT
            )

        return True

    except Exception as error:

        st.error(
            f"Could not open {script_name}: {error}"
        )

        return False


# ============================================================
# OPEN STREAMLIT PAGE
# ============================================================

def open_streamlit_page(script_name):
    """
    Opens another CEMS Streamlit page,
    such as Session Management.
    """

    script_path = os.path.join(
        PROJECT_ROOT,
        script_name
    )

    if not os.path.exists(script_path):

        st.error(
            f"{script_name} could not be found."
        )

        return False

    try:

        python_executable = get_python_executable()

        if os.name == "nt":

            subprocess.Popen(
                [
                    python_executable,
                    "-m",
                    "streamlit",
                    "run",
                    script_path
                ],
                cwd=PROJECT_ROOT,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

        else:

            subprocess.Popen(
                [
                    python_executable,
                    "-m",
                    "streamlit",
                    "run",
                    script_path
                ],
                cwd=PROJECT_ROOT
            )

        return True

    except Exception as error:

        st.error(
            f"Could not open {script_name}: {error}"
        )

        return False


# ============================================================
# CONTROL CENTRE
# ============================================================

st.subheader("🎛️ Control Centre")

control1, control2, control3, control4, control5 = st.columns(5)


# ------------------------------------------------------------
# SESSION MANAGEMENT
# ------------------------------------------------------------

with control1:

    if st.button(
        "📚 Start / End Session",
        use_container_width=True
    ):

        opened = open_streamlit_page(
            "dashboard/session_management.py"
        )

        if opened:

            st.success(
                "Session Management opened."
            )


# ------------------------------------------------------------
# FACE REGISTRATION
# ------------------------------------------------------------

with control2:

    if st.button(
        "👤 Register Student Face",
        use_container_width=True
    ):

        started = open_program(
            "face_registration.py"
        )

        if started:

            st.success(
                "Face Registration opened. "
                "Complete the registration in the new window."
            )


# ------------------------------------------------------------
# LIVE MONITORING
# ------------------------------------------------------------

with control3:

    if st.button(
        "🎥 Live Monitoring",
        use_container_width=True
    ):

        started = open_program(
            "tracking.py"
        )

        if started:

            st.success(
                "Live Monitoring opened. "
                "Choose 1 for USB Camera."
            )


# ------------------------------------------------------------
# ANALYTICS
# ------------------------------------------------------------

with control4:

    if st.button(
        "📊 Analytics",
        use_container_width=True
    ):

        st.session_state["show_analytics"] = (
            not st.session_state["show_analytics"]
        )

        st.rerun()


# ------------------------------------------------------------
# REFRESH
# ------------------------------------------------------------

with control5:

    if st.button(
        "🔄 Refresh",
        use_container_width=True
    ):

        st.rerun()


st.divider()


# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_PATH):

    st.error(
        "CEMS database was not found."
    )

    st.stop()


try:

    connection = sqlite3.connect(
        DB_PATH
    )

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

    st.error(
        f"Could not load database: {error}"
    )

    st.stop()


# ============================================================
# CHECK DATA
# ============================================================

if data.empty:

    st.warning(
        "No engagement records are available yet. "
        "Start Live Monitoring to collect data."
    )

    st.stop()


data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    errors="coerce"
)

data = data.dropna(
    subset=["timestamp"]
)

data["date"] = (
    data["timestamp"].dt.date
)


# ============================================================
# STUDENT NAMES
# ============================================================

student_names = {}


if not students.empty:

    for _, student in students.iterrows():

        student_id = student.iloc[0]

        if len(student) >= 3:

            student_name = student.iloc[2]

            student_names[
                int(student_id)
            ] = student_name


def get_student_name(student_id):

    if pd.isna(student_id):

        return "Unregistered"

    student_id = int(
        student_id
    )

    return student_names.get(
        student_id,
        f"Student {student_id}"
    )


# ============================================================
# ANALYTICS
# ============================================================

if st.session_state["show_analytics"]:

    st.header(
        "📊 Analytics"
    )

    st.caption(
        "Daily, weekly and low-engagement analytics."
    )

    try:

        from analytics import get_analytics
        from alerts import get_students_requiring_attention

        analytics_data = get_analytics()


        # ----------------------------------------------------
        # ANALYTICS SUMMARY
        # ----------------------------------------------------

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Overall Average",
            f"{analytics_data['average_engagement']:.1f}%"
        )

        a2.metric(
            "Total Records",
            analytics_data["total_records"]
        )

        a3.metric(
            "Low Engagement",
            analytics_data["low_engagement_count"]
        )


        # ----------------------------------------------------
        # DAILY AVERAGE
        # ----------------------------------------------------

        st.subheader(
            "📅 Daily Average"
        )

        daily = analytics_data.get(
            "daily_averages",
            {}
        )

        if daily:

            daily_df = pd.DataFrame(
                [
                    {
                        "Date": date,
                        "Average Engagement (%)": score
                    }
                    for date, score in daily.items()
                ]
            )

            daily_df[
                "Average Engagement (%)"
            ] = daily_df[
                "Average Engagement (%)"
            ].round(1)

            st.dataframe(
                daily_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No daily engagement data available."
            )


        # ----------------------------------------------------
        # WEEKLY AVERAGE
        # ----------------------------------------------------

        st.subheader(
            "📆 Weekly Average"
        )

        weekly = analytics_data.get(
            "weekly_averages",
            {}
        )

        if weekly:

            weekly_df = pd.DataFrame(
                [
                    {
                        "Week": week,
                        "Average Engagement (%)": score
                    }
                    for week, score in weekly.items()
                ]
            )

            weekly_df[
                "Average Engagement (%)"
            ] = weekly_df[
                "Average Engagement (%)"
            ].round(1)

            st.dataframe(
                weekly_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No weekly engagement data available."
            )


        # ----------------------------------------------------
        # SUSTAINED LOW ENGAGEMENT
        # ----------------------------------------------------

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


        st.divider()

    except Exception as error:

        st.error(
            f"Could not load analytics: {error}"
        )


# ============================================================
# FILTERS
# ============================================================

st.header(
    "🔎 View Engagement"
)

filter1, filter2, filter3 = st.columns(3)

filtered = data.copy()


# ------------------------------------------------------------
# SESSION FILTER
# ------------------------------------------------------------

session_values = sorted(
    data["session_id"]
    .dropna()
    .unique()
    .tolist()
)


with filter1:

    selected_session = st.selectbox(
        "Session",
        ["All"] + session_values
    )


if selected_session != "All":

    filtered = filtered[
        filtered["session_id"]
        == selected_session
    ]


# ------------------------------------------------------------
# DATE FILTER
# ------------------------------------------------------------

date_values = sorted(
    filtered["date"]
    .dropna()
    .unique()
    .tolist()
)


with filter2:

    selected_date = st.selectbox(
        "Date",
        ["All"] + date_values
    )


if selected_date != "All":

    filtered = filtered[
        filtered["date"]
        == selected_date
    ]


# ------------------------------------------------------------
# STUDENT FILTER
# ------------------------------------------------------------

registered_ids = sorted(
    filtered["registered_student_id"]
    .dropna()
    .unique()
    .tolist()
)


student_options = {
    "All Students": None
}


for student_id in registered_ids:

    student_id = int(
        student_id
    )

    student_name = get_student_name(
        student_id
    )

    student_options[
        f"{student_name} (ID {student_id})"
    ] = student_id


with filter3:

    selected_student_label = st.selectbox(
        "Student",
        list(
            student_options.keys()
        )
    )


selected_student = (
    student_options[
        selected_student_label
    ]
)


if selected_student is not None:

    filtered = filtered[
        filtered[
            "registered_student_id"
        ]
        == selected_student
    ]


# ============================================================
# CHECK FILTER RESULTS
# ============================================================

if filtered.empty:

    st.warning(
        "No records match the selected filters."
    )

    st.stop()


# ============================================================
# ENGAGEMENT SUMMARY
# ============================================================

st.divider()

st.header(
    "📌 Engagement Summary"
)


average_engagement = (
    filtered[
        "engagement_score"
    ].mean()
)


registered_student_count = (
    filtered[
        "registered_student_id"
    ]
    .dropna()
    .nunique()
)


low_engagement = filtered[
    filtered[
        "engagement_score"
    ] < 60
]


m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "Average Engagement",
    f"{average_engagement:.1f}%"
)


m2.metric(
    "Students",
    registered_student_count
)


m3.metric(
    "Records",
    len(filtered)
)


m4.metric(
    "Low Engagement",
    len(low_engagement)
)


# ============================================================
# SIMPLE STUDENT GRAPH
# ============================================================

st.divider()

st.header(
    "👥 Average Engagement by Student"
)

st.write(
    "This graph compares the average engagement "
    "of each registered student."
)


student_average = (
    filtered
    .dropna(
        subset=[
            "registered_student_id"
        ]
    )
    .groupby(
        "registered_student_id"
    )["engagement_score"]
    .mean()
)


if student_average.empty:

    st.info(
        "No registered student engagement data available."
    )

else:

    student_average.index = [
        get_student_name(
            student_id
        )
        for student_id
        in student_average.index
    ]

    student_average = (
        student_average.round(1)
    )


    student_df = (
        student_average
        .reset_index()
    )

    student_df.columns = [
        "Student",
        "Engagement (%)"
    ]


    # --------------------------------------------------------
    # BAR GRAPH
    # --------------------------------------------------------

    st.bar_chart(
        student_df,
        x="Student",
        y="Engagement (%)",
        y_label="Engagement %"
    )


    # --------------------------------------------------------
    # EASY RESULTS
    # --------------------------------------------------------

    st.subheader(
        "Student Results"
    )


    for _, row in student_df.iterrows():

        student_name = (
            row["Student"]
        )

        score = (
            row["Engagement (%)"]
        )


        if score < 60:

            st.warning(
                f"⚠️ {student_name}: "
                f"{score:.1f}% — Low Engagement"
            )

        else:

            st.success(
                f"✅ {student_name}: "
                f"{score:.1f}%"
            )


# ============================================================
# ENGAGEMENT RECORDS
# ============================================================

st.divider()

st.header(
    "📋 Engagement Records"
)


records = filtered[
    [
        "timestamp",
        "registered_student_id",
        "session_id",
        "engagement_score",
        "status",
        "orientation"
    ]
].copy()


records["Student"] = (
    records[
        "registered_student_id"
    ].apply(
        get_student_name
    )
)


records = records[
    [
        "timestamp",
        "Student",
        "session_id",
        "engagement_score",
        "status",
        "orientation"
    ]
]


records.columns = [
    "Time",
    "Student",
    "Session",
    "Engagement (%)",
    "Status",
    "Direction"
]


st.dataframe(
    records,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# LOW ENGAGEMENT RECORDS
# ============================================================

st.divider()

st.header(
    "🚨 Low Engagement Records"
)

st.caption(
    "An engagement score below 60% "
    "is considered low engagement."
)


if low_engagement.empty:

    st.success(
        "✅ No low-engagement records "
        "for the selected filters."
    )

else:

    st.warning(
        f"⚠️ {len(low_engagement)} "
        "low-engagement record(s) detected."
    )


    alert_table = low_engagement[
        [
            "timestamp",
            "registered_student_id",
            "engagement_score"
        ]
    ].copy()


    alert_table["Student"] = (
        alert_table[
            "registered_student_id"
        ].apply(
            get_student_name
        )
    )


    alert_table = alert_table[
        [
            "timestamp",
            "Student",
            "engagement_score"
        ]
    ]


    alert_table.columns = [
        "Time",
        "Student",
        "Engagement (%)"
    ]


    st.dataframe(
        alert_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DEMO EXPLANATION
# ============================================================

st.divider()


with st.expander(
    "ℹ️ How to explain this dashboard"
):

    st.markdown(
        """
### CEMS Workflow

**1. Start / End Session**  
Opens Session Management where a classroom session
can be started or ended.

**2. Register Student Face**  
Registers a student's face so the system can recognise
the student during monitoring.

**3. Live Monitoring**  
Starts the camera and engagement monitoring system.
Choose **1** for the USB camera.

**4. Analytics**  
Shows overall engagement, daily and weekly averages,
and sustained low-engagement alerts.

**5. Average Engagement by Student**  
Compares the average engagement of each registered
student.

**6. Low Engagement**  
An engagement score below **60%** is treated as
low engagement.
        """
    )