import streamlit as st

from database import (
    get_units,
    get_classrooms,
    start_session,
    end_session,
    get_sessions
)


st.set_page_config(
    page_title="CEMS - Session Management",
    page_icon="📚",
    layout="wide"
)

st.title("CEMS Session Management")

st.write(
    "Start and end classroom monitoring sessions by selecting a unit and classroom."
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

units = get_units()
classrooms = get_classrooms()


# -------------------------------------------------
# START SESSION
# -------------------------------------------------

st.header("Start New Session")

if len(units) == 0:
    st.warning("Please add at least one unit first.")

elif len(classrooms) == 0:
    st.warning("Please add at least one classroom first.")

else:
    unit_options = {
        f"{unit[1]} - {unit[2]}": unit[0]
        for unit in units
    }

    classroom_options = {
        classroom[1]: classroom[0]
        for classroom in classrooms
    }

    selected_unit = st.selectbox(
        "Select Unit",
        list(unit_options.keys())
    )

    selected_classroom = st.selectbox(
        "Select Classroom",
        list(classroom_options.keys())
    )

    if st.button("Start Session"):

        unit_id = unit_options[selected_unit]
        classroom_id = classroom_options[selected_classroom]

        session_id = start_session(
            unit_id,
            classroom_id
        )

        st.success(
            f"Session started successfully. Session ID: {session_id}"
        )


st.divider()


# -------------------------------------------------
# SESSION HISTORY
# -------------------------------------------------

st.header("Session History")

sessions = get_sessions()

if len(sessions) == 0:
    st.info("No classroom sessions have been created yet.")

else:

    for session in sessions:

        session_id = session[0]
        unit_code = session[1]
        unit_name = session[2]
        classroom_name = session[3]
        start_time = session[4]
        end_time = session[5]

        st.subheader(
            f"Session {session_id} - {unit_code}"
        )

        st.write(
            f"**Unit:** {unit_code} - {unit_name}"
        )

        st.write(
            f"**Classroom:** {classroom_name}"
        )

        st.write(
            f"**Started:** {start_time}"
        )

        if end_time is None:

            st.warning("Session is currently active.")

            if st.button(
                "End Session",
                key=f"end_session_{session_id}"
            ):

                end_session(session_id)

                st.success(
                    f"Session {session_id} ended successfully."
                )

                st.rerun()

        else:

            st.write(
                f"**Ended:** {end_time}"
            )

            st.success("Session completed.")

        st.divider()