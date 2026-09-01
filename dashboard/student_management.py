import streamlit as st

from database import add_student, get_students


st.set_page_config(
    page_title="CEMS - Student Management",
    page_icon="🎓",
    layout="wide"
)

st.title("CEMS Student Management")

st.write(
    "Register students so engagement records can later be linked "
    "to real student information."
)

st.subheader("Add Student")

student_number = st.text_input(
    "Student Number",
    placeholder="Example: CIHE240247"
)

student_name = st.text_input(
    "Student Name",
    placeholder="Example: Student Name"
)

if st.button("Add Student"):

    if student_number.strip() == "" or student_name.strip() == "":
        st.warning("Please enter both student number and student name.")

    else:
        result = add_student(
            student_number.strip(),
            student_name.strip()
        )

        if result:
            st.success("Student registered successfully.")
        else:
            st.error(
                "This student number already exists."
            )


st.divider()

st.subheader("Registered Students")

students = get_students()

if len(students) == 0:
    st.info("No students have been registered yet.")

else:
    for student in students:

        database_id = student[0]
        student_number = student[1]
        student_name = student[2]

        st.write(
            f"**{student_number}** — {student_name}"
        )