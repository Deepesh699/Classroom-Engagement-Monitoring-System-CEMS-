from datetime import datetime
from database import get_all_records

# Index positions returned by database.get_all_records()
TIMESTAMP = 1
ENGAGEMENT_SCORE = 3
STATUS = 4
REGISTERED_STUDENT_ID = 5
SESSION_ID = 6
ORIENTATION = 7

LOW_THRESHOLD = 60


def calculate_average(records):
    """Calculate average engagement score."""
    if not records:
        return 0

    scores = [float(record[ENGAGEMENT_SCORE]) for record in records]

    return round(sum(scores) / len(scores), 2)


def parse_timestamp(timestamp):
    """Convert a database timestamp into a datetime object."""
    if isinstance(timestamp, datetime):
        return timestamp

    try:
        return datetime.fromisoformat(str(timestamp))
    except (ValueError, TypeError):
        return None


def get_student_average(registered_student_id):
    """
    Return average engagement for an actual registered student.

    Important:
    Uses registered_student_id, NOT student_id.
    student_id is only the temporary camera tracking ID.
    """
    records = get_all_records()

    student_records = [
        record for record in records
        if record[REGISTERED_STUDENT_ID] == registered_student_id
    ]

    return calculate_average(student_records)


def get_session_average(session_id):
    """Return average engagement for one session."""
    records = get_all_records()

    session_records = [
        record for record in records
        if record[SESSION_ID] == session_id
    ]

    return calculate_average(session_records)


def get_daily_engagement_averages():
    """Return average engagement grouped by calendar date."""
    records = get_all_records()

    daily_records = {}

    for record in records:
        timestamp = parse_timestamp(record[TIMESTAMP])

        if timestamp is None:
            continue

        date_key = timestamp.date().isoformat()

        if date_key not in daily_records:
            daily_records[date_key] = []

        daily_records[date_key].append(record)

    return {
        date_key: calculate_average(records_for_day)
        for date_key, records_for_day in sorted(daily_records.items())
    }


def get_weekly_engagement_averages():
    """Return average engagement grouped by ISO calendar week."""
    records = get_all_records()

    weekly_records = {}

    for record in records:
        timestamp = parse_timestamp(record[TIMESTAMP])

        if timestamp is None:
            continue

        iso_year, iso_week, _ = timestamp.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"

        if week_key not in weekly_records:
            weekly_records[week_key] = []

        weekly_records[week_key].append(record)

    return {
        week_key: calculate_average(records_for_week)
        for week_key, records_for_week in sorted(weekly_records.items())
    }


def get_low_engagement_count(
    registered_student_id=None,
    session_id=None,
    threshold=LOW_THRESHOLD
):
    """Count engagement records below the chosen threshold."""
    records = get_all_records()

    filtered_records = []

    for record in records:

        if (
            registered_student_id is not None
            and record[REGISTERED_STUDENT_ID] != registered_student_id
        ):
            continue

        if (
            session_id is not None
            and record[SESSION_ID] != session_id
        ):
            continue

        if float(record[ENGAGEMENT_SCORE]) < threshold:
            filtered_records.append(record)

    return len(filtered_records)


def get_engagement_trend(
    registered_student_id=None,
    session_id=None
):
    """
    Return engagement scores over time.

    Can be filtered by registered student and/or session.
    """
    records = get_all_records()

    trend = []

    for record in records:

        if (
            registered_student_id is not None
            and record[REGISTERED_STUDENT_ID] != registered_student_id
        ):
            continue

        if (
            session_id is not None
            and record[SESSION_ID] != session_id
        ):
            continue

        trend.append({
            "timestamp": record[TIMESTAMP],
            "engagement_score": record[ENGAGEMENT_SCORE],
            "status": record[STATUS],
            "orientation": record[ORIENTATION]
        })

    # database.py returns newest records first.
    # Reverse so trend is oldest -> newest.
    trend.reverse()

    return trend


def get_session_comparison():
    """Return average engagement for each session."""
    records = get_all_records()

    session_ids = sorted({
        record[SESSION_ID]
        for record in records
        if record[SESSION_ID] is not None
    })

    comparison = {}

    for session_id in session_ids:

        session_records = [
            record for record in records
            if record[SESSION_ID] == session_id
        ]

        comparison[session_id] = calculate_average(
            session_records
        )

    return comparison


def get_student_comparison():
    """Return average engagement for each registered student."""
    records = get_all_records()

    student_ids = sorted({
        record[REGISTERED_STUDENT_ID]
        for record in records
        if record[REGISTERED_STUDENT_ID] is not None
    })

    comparison = {}

    for registered_student_id in student_ids:

        student_records = [
            record for record in records
            if record[REGISTERED_STUDENT_ID]
            == registered_student_id
        ]

        comparison[registered_student_id] = calculate_average(
            student_records
        )

    return comparison


def get_analytics():
    """Return overall classroom analytics summary."""
    records = get_all_records()

    return {
        "average_engagement": calculate_average(records),
        "total_records": len(records),
        "low_engagement_count": get_low_engagement_count(),
        "student_comparison": get_student_comparison(),
        "session_comparison": get_session_comparison(),
        "daily_averages": get_daily_engagement_averages(),
        "weekly_averages": get_weekly_engagement_averages()
    }


if __name__ == "__main__":
    print("CEMS Engagement Analytics")
    print(get_analytics())