import streamlit as st

from database import (
    get_students,
    get_sessions,
    save_engagement
)

st.set_page_config(
    page_title="CEMS - Engagement Assignment",
    page_icon="📊",
    layout="wide"
)

st.title("CEMS Engagement Assignment")

st.write(
    "Link engagement records to registered students and classroom sessions."
)

students = get_students()
sessions = get_sessions()

if len(students) == 0:
    st.warning("No registered students found.")

elif len(sessions) == 0:
    st.warning("No classroom sessions found.")

else:

    student_options = {
        f"{student[1]} - {student[2]}": student[0]
        for student in students
    }

    session_options = {
        f"Session {session[0]} - {session[1]} - {session[3]}": session[0]
        for session in sessions
    }

    selected_student = st.selectbox(
        "Select Student",
        list(student_options.keys())
    )

    selected_session = st.selectbox(
        "Select Session",
        list(session_options.keys())
    )

    tracking_id = st.number_input(
        "Temporary Tracking ID",
        min_value=1,
        value=1,
        step=1
    )

    score = st.slider(
        "Engagement Score",
        min_value=0,
        max_value=100,
        value=80
    )

    if score >= 70:
        status = "Engaged"
    elif score >= 50:
        status = "Neutral"
    else:
        status = "Low"

    st.write(f"Status: **{status}**")

    if st.button("Save Engagement Record"):

        registered_student_id = student_options[selected_student]
        session_id = session_options[selected_session]

        save_engagement(
            tracking_id,
            score,
            status,
            registered_student_id,
            session_id
        )

        st.success(
            "Engagement record saved successfully."
        )