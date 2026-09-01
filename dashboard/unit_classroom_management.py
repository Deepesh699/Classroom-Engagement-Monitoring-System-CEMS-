import streamlit as st

from database import (
    add_unit,
    get_units,
    add_classroom,
    get_classrooms
)


st.set_page_config(
    page_title="CEMS - Units and Classrooms",
    page_icon="🏫",
    layout="wide"
)

st.title("CEMS Unit and Classroom Management")

st.write(
    "Manage teaching units and classrooms before creating classroom monitoring sessions."
)

# -------------------------------------------------
# UNIT MANAGEMENT
# -------------------------------------------------

st.header("Unit Management")

unit_code = st.text_input(
    "Unit Code",
    placeholder="Example: ICT308"
)

unit_name = st.text_input(
    "Unit Name",
    placeholder="Example: Project 2"
)

if st.button("Add Unit"):

    if unit_code.strip() == "" or unit_name.strip() == "":
        st.warning("Please enter both unit code and unit name.")

    else:
        result = add_unit(
            unit_code.strip().upper(),
            unit_name.strip()
        )

        if result:
            st.success("Unit added successfully.")
        else:
            st.error("This unit code already exists.")


st.subheader("Registered Units")

units = get_units()

if len(units) == 0:
    st.info("No units have been added yet.")

else:
    for unit in units:

        unit_id = unit[0]
        unit_code = unit[1]
        unit_name = unit[2]

        st.write(
            f"**{unit_code}** — {unit_name}"
        )


st.divider()

# -------------------------------------------------
# CLASSROOM MANAGEMENT
# -------------------------------------------------

st.header("Classroom Management")

classroom_name = st.text_input(
    "Classroom Name",
    placeholder="Example: Room 201"
)

if st.button("Add Classroom"):

    if classroom_name.strip() == "":
        st.warning("Please enter a classroom name.")

    else:
        result = add_classroom(
            classroom_name.strip()
        )

        if result:
            st.success("Classroom added successfully.")
        else:
            st.error("This classroom already exists.")


st.subheader("Registered Classrooms")

classrooms = get_classrooms()

if len(classrooms) == 0:
    st.info("No classrooms have been added yet.")

else:
    for classroom in classrooms:

        classroom_id = classroom[0]
        classroom_name = classroom[1]

        st.write(
            f"**{classroom_name}**"
        )