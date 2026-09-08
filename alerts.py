ALERT_THRESHOLD = 60
SUSTAINED_LOW_COUNT = 3


def check_alert(student_id, score):
    if float(score) < ALERT_THRESHOLD:
        return {
            "alert": True,
            "student_id": student_id,
            "score": score,
            "message":
                f"Student {student_id} engagement is low ({score}%)."
        }

    return {
        "alert": False,
        "student_id": student_id,
        "score": score,
        "message": ""
    }


def check_sustained_low_engagement(student_id, scores):
    if len(scores) < SUSTAINED_LOW_COUNT:
        return {
            "alert": False,
            "message": ""
        }

    recent_scores = scores[-SUSTAINED_LOW_COUNT:]

    sustained_low = all(
        float(score) < ALERT_THRESHOLD
        for score in recent_scores
    )

    if sustained_low:
        return {
            "alert": True,
            "message":
                f"Student {student_id} has sustained low engagement."
        }

    return {
        "alert": False,
        "message": ""
    }


if __name__ == "__main__":
    print(check_alert("Student-1", 45))
    print(
        check_sustained_low_engagement(
            "Student-1",
            [75, 55, 48, 42]
        )
    )