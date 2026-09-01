import streamlit as st

from database import (
    get_students,
    get_sessions,
    assign_track_to_student,
    get_track_assignments
)

st.set_page_config(
    page_title="CEMS - Track Assignment",
    page_icon="🔗",
    layout="wide"
)

st.title("CEMS Track-to-Student Assignment")

st.write(
    "Assign temporary camera tracking IDs to registered students "
    "for a classroom session."
)

students = get_students()
sessions = get_sessions()

if len(students) == 0:
    st.warning("No registered students available.")

elif len(sessions) == 0:
    st.warning("No classroom sessions available.")

else:
    session_options = {
        f"Session {session[0]} - {session[1]} - {session[3]}": session[0]
        for session in sessions
    }

    student_options = {
        f"{student[1]} - {student[2]}": student[0]
        for student in students
    }

    selected_session = st.selectbox(
        "Select Session",
        list(session_options.keys())
    )

    selected_student = st.selectbox(
        "Select Student",
        list(student_options.keys())
    )

    track_id = st.number_input(
        "Temporary Tracking ID",
        min_value=1,
        value=1,
        step=1
    )

    if st.button("Assign Track"):
        session_id = session_options[selected_session]
        registered_student_id = student_options[selected_student]

        result = assign_track_to_student(
            session_id,
            track_id,
            registered_student_id
        )

        if result:
            st.success(
                f"Track {track_id} assigned successfully."
            )
        else:
            st.error(
                "This tracking ID is already assigned in this session."
            )

    st.divider()

    st.subheader("Current Assignments")

    selected_session_id = session_options[selected_session]

    assignments = get_track_assignments(
        selected_session_id
    )

    if len(assignments) == 0:
        st.info(
            "No tracking IDs have been assigned for this session."
        )

    else:
        for assignment in assignments:
            track_id = assignment[1]
            student_number = assignment[3]
            student_name = assignment[4]

            st.write(
                f"**Track {track_id}** → "
                f"{student_number} - {student_name}"
            )