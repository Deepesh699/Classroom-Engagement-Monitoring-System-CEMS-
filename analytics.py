from storage import (
    get_all_records,
    get_student_records,
    get_session_records,
    get_student_session_records
)

LOW_THRESHOLD = 60


def calculate_average(records):
    if not records:
        return 0

    scores = [
        float(record["engagement_score"])
        for record in records
    ]

    return round(sum(scores) / len(scores), 2)


def get_low_engagement_count(records):
    return sum(
        1
        for record in records
        if float(record["engagement_score"]) < LOW_THRESHOLD
    )


def get_student_average(registered_student_id):
    records = get_student_records(registered_student_id)

    return calculate_average(records)


def get_session_average(session_id):
    records = get_session_records(session_id)

    return calculate_average(records)


def get_student_session_average(registered_student_id, session_id):
    records = get_student_session_records(
        registered_student_id,
        session_id
    )

    return calculate_average(records)


def get_analytics():
    records = get_all_records()

    return {
        "average_engagement": calculate_average(records),
        "records": len(records),
        "low_engagement_records": get_low_engagement_count(records)
    }


if __name__ == "__main__":
    print(get_analytics())