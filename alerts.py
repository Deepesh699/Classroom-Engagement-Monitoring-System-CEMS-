from database import get_all_records

ALERT_THRESHOLD = 60
SUSTAINED_LOW_COUNT = 3

ENGAGEMENT_SCORE = 3
REGISTERED_STUDENT_ID = 5
SESSION_ID = 6


def check_alert(student_id, score, threshold=ALERT_THRESHOLD):
    """Check a single engagement score."""
    score = float(score)

    if score < threshold:
        return {
            "alert": True,
            "message":
                f"Student {student_id} engagement is low ({score}%)."
        }

    return {
        "alert": False,
        "message": ""
    }


def check_sustained_low_engagement(
    registered_student_id,
    session_id=None,
    threshold=ALERT_THRESHOLD,
    required_count=SUSTAINED_LOW_COUNT
):
    """
    Check repeated low engagement for an actual registered student.
    Uses registered_student_id, not the temporary tracking student_id.
    """

    records = get_all_records()

    student_records = [
        record for record in records
        if record[REGISTERED_STUDENT_ID] == registered_student_id
    ]

    if session_id is not None:
        student_records = [
            record for record in student_records
            if record[SESSION_ID] == session_id
        ]

    # database.py returns newest records first
    recent_records = student_records[:required_count]

    if len(recent_records) < required_count:
        return {
            "alert": False,
            "message": ""
        }

    sustained_low = all(
        float(record[ENGAGEMENT_SCORE]) < threshold
        for record in recent_records
    )

    if sustained_low:
        return {
            "alert": True,
            "message":
                f"Registered student {registered_student_id} "
                f"has sustained low engagement."
        }

    return {
        "alert": False,
        "message": ""
    }


def get_students_requiring_attention(
    session_id=None,
    threshold=ALERT_THRESHOLD
):
    """Return registered students showing sustained low engagement."""

    records = get_all_records()

    registered_students = sorted({
        record[REGISTERED_STUDENT_ID]
        for record in records
        if record[REGISTERED_STUDENT_ID] is not None
    })

    alerts = []

    for registered_student_id in registered_students:

        result = check_sustained_low_engagement(
            registered_student_id,
            session_id=session_id,
            threshold=threshold
        )

        if result["alert"]:
            alerts.append({
                "registered_student_id": registered_student_id,
                "message": result["message"]
            })

    return alerts


if __name__ == "__main__":
    print("Single low-score test:")
    print(check_alert("Track-1", 45))

    print("\nStudents requiring attention:")
    print(get_students_requiring_attention())