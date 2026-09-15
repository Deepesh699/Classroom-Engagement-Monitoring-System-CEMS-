from database import get_student_by_id, get_active_session_id


def verify_recognised_student(recognition_result):
    """
    Verify that a recognised face belongs to a real
    registered student stored in the CEMS SQLite database.

    SQLite is treated as the authoritative source for
    student name and student number.
    """

    if not recognition_result.get("recognised"):
        return {
            "valid": False,
            "reason": "Unknown face",
            "student": None,
            "session_id": None
        }

    student_id = recognition_result.get("student_id")

    if student_id is None:
        return {
            "valid": False,
            "reason": "Recognition result has no student ID",
            "student": None,
            "session_id": None
        }

    student = get_student_by_id(student_id)

    if student is None:
        return {
            "valid": False,
            "reason": "Recognised student does not exist in database",
            "student": None,
            "session_id": None
        }

    active_session_id = get_active_session_id()

    return {
        "valid": True,
        "reason": "Student verified",
        "student": {
            "id": student[0],
            "student_number": student[1],
            "student_name": student[2],
            "created_at": student[3]
        },
        "session_id": active_session_id
    }