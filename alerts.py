ALERT_THRESHOLD = 60
def check_alert(student_id, score):
    if score < ALERT_THRESHOLD:
        return {
            "alert": True,
            "message":
                f"Student {student_id} engagement is low ({score}%)."
        }
    return {
        "alert": False,
        "message": ""
    }