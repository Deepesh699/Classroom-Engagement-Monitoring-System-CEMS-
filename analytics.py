from storage import get_all_records
from datetime import datetime, timedelta

LOW_THRESHOLD = 60


def _get_scores(records):
    return [
        float(record["engagement_score"])
        for record in records
    ]


def calculate_average(records):
    scores = _get_scores(records)

    if not scores:
        return 0

    return round(sum(scores) / len(scores), 2)


def get_student_average(student_id):
    records = get_all_records()

    student_records = [
        record for record in records
        if str(record["student_id"]) == str(student_id)
    ]

    return calculate_average(student_records)


def get_daily_average(date=None):
    records = get_all_records()

    if date is None:
        date = datetime.now().date()

    daily_records = []

    for record in records:
        try:
            record_date = datetime.fromisoformat(
                record["timestamp"]
            ).date()

            if record_date == date:
                daily_records.append(record)
        except (ValueError, TypeError):
            continue

    return calculate_average(daily_records)


def get_weekly_average():
    records = get_all_records()

    start_date = datetime.now() - timedelta(days=7)
    weekly_records = []

    for record in records:
        try:
            timestamp = datetime.fromisoformat(
                record["timestamp"]
            )

            if timestamp >= start_date:
                weekly_records.append(record)
        except (ValueError, TypeError):
            continue

    return calculate_average(weekly_records)


def get_classroom_average():
    records = get_all_records()
    return calculate_average(records)


def get_low_engagement_count():
    records = get_all_records()

    return sum(
        1 for record in records
        if float(record["engagement_score"]) < LOW_THRESHOLD
    )


def get_analytics():
    records = get_all_records()

    return {
        "average_engagement": calculate_average(records),
        "records": len(records),
        "low_engagement_records": get_low_engagement_count(),
        "daily_average": get_daily_average(),
        "weekly_average": get_weekly_average(),
        "classroom_average": get_classroom_average()
    }


if __name__ == "__main__":
    print("CEMS Engagement Analytics")
    print(get_analytics())