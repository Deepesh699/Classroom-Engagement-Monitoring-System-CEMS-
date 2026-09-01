ALERT_THRESHOLD = 60


def check_alert(student_id, score):
    if score < ALERT_THRESHOLD:
        return {
            "alert": True,
            "message": f"Student {student_id} engagement is low ({score}%)."
        }

    return {
        "alert": False,
        "message": ""
    }


if __name__ == "__main__":
    print(check_alert(1, 50))
    print(check_alert(2, 80))