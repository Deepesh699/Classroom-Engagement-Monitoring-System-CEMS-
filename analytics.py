from collections import defaultdict
from datetime import datetime

from database import get_all_records


# ============================================================
# HELPERS
# ============================================================

def calculate_average(values):
    """Return average rounded to two decimal places."""

    if not values:
        return 0.0

    return round(
        sum(values) / len(values),
        2
    )


def parse_timestamp(timestamp):
    """Convert database timestamp into datetime."""

    if isinstance(timestamp, datetime):
        return timestamp

    try:
        return datetime.fromisoformat(str(timestamp))
    except (ValueError, TypeError):
        return None


# ============================================================
# FILTER RECORDS
# ============================================================

def get_filtered_records(session_id=None):
    """
    Return engagement records.

    If session_id is provided, only records belonging
    to that classroom session are returned.
    """

    records = get_all_records()

    if session_id is None:
        return records

    return [
        record
        for record in records
        if record[6] == session_id
    ]


# ============================================================
# STUDENT AVERAGE
# ============================================================

def get_student_average(
    registered_student_id,
    session_id=None
):

    records = get_filtered_records(
        session_id
    )

    scores = [
        record[3]
        for record in records
        if record[5] == registered_student_id
    ]

    return calculate_average(scores)


# ============================================================
# SESSION AVERAGE
# ============================================================

def get_session_average(session_id):

    records = get_filtered_records(
        session_id
    )

    scores = [
        record[3]
        for record in records
    ]

    return calculate_average(scores)


# ============================================================
# DAILY AVERAGES
# ============================================================

def get_daily_engagement_averages(
    session_id=None
):

    records = get_filtered_records(
        session_id
    )

    daily_scores = defaultdict(list)

    for record in records:

        timestamp = parse_timestamp(
            record[1]
        )

        if timestamp is None:
            continue

        date_key = (
            timestamp
            .date()
            .isoformat()
        )

        daily_scores[
            date_key
        ].append(
            record[3]
        )

    return {
        date: calculate_average(scores)
        for date, scores
        in sorted(daily_scores.items())
    }


# ============================================================
# WEEKLY AVERAGES
# ============================================================

def get_weekly_engagement_averages(
    session_id=None
):

    records = get_filtered_records(
        session_id
    )

    weekly_scores = defaultdict(list)

    for record in records:

        timestamp = parse_timestamp(
            record[1]
        )

        if timestamp is None:
            continue

        iso_year, iso_week, _ = (
            timestamp.isocalendar()
        )

        week_key = (
            f"{iso_year}-W{iso_week:02d}"
        )

        weekly_scores[
            week_key
        ].append(
            record[3]
        )

    return {
        week: calculate_average(scores)
        for week, scores
        in sorted(weekly_scores.items())
    }


# ============================================================
# LOW ENGAGEMENT COUNT
# ============================================================

def get_low_engagement_count(
    session_id=None,
    threshold=60
):

    records = get_filtered_records(
        session_id
    )

    return sum(
        1
        for record in records
        if record[3] < threshold
    )


# ============================================================
# ENGAGEMENT TREND
# ============================================================

def get_engagement_trend(
    registered_student_id,
    session_id=None
):

    records = get_filtered_records(
        session_id
    )

    trend = []

    for record in records:

        if (
            record[5]
            == registered_student_id
        ):

            trend.append(
                {
                    "timestamp": record[1],
                    "engagement_score": record[3],
                    "status": record[4]
                }
            )

    trend.sort(
        key=lambda item:
        str(item["timestamp"])
    )

    return trend


# ============================================================
# SESSION COMPARISON
# ============================================================

def get_session_comparison():

    records = get_all_records()

    session_scores = defaultdict(list)

    for record in records:

        session_id = record[6]

        if session_id is None:
            continue

        session_scores[
            session_id
        ].append(
            record[3]
        )

    return {
        session_id:
        calculate_average(scores)

        for session_id, scores
        in sorted(session_scores.items())
    }


# ============================================================
# STUDENT COMPARISON
# ============================================================

def get_student_comparison(
    session_id=None
):

    records = get_filtered_records(
        session_id
    )

    student_scores = defaultdict(list)

    for record in records:

        registered_student_id = (
            record[5]
        )

        if registered_student_id is None:
            continue

        student_scores[
            registered_student_id
        ].append(
            record[3]
        )

    return {
        student_id:
        calculate_average(scores)

        for student_id, scores
        in sorted(student_scores.items())
    }


# ============================================================
# MAIN ANALYTICS
# ============================================================

def get_analytics(session_id=None):
    """
    Return analytics for all records or one session.

    Example:
        get_analytics()
            -> all database records

        get_analytics(5)
            -> Session 5 only
    """

    records = get_filtered_records(
        session_id
    )

    scores = [
        record[3]
        for record in records
    ]

    return {

        "session_id":
            session_id,

        "average_engagement":
            calculate_average(scores),

        "total_records":
            len(records),

        "low_engagement_count":
            get_low_engagement_count(
                session_id
            ),

        "student_comparison":
            get_student_comparison(
                session_id
            ),

        "session_comparison":
            get_session_comparison(),

        "daily_averages":
            get_daily_engagement_averages(
                session_id
            ),

        "weekly_averages":
            get_weekly_engagement_averages(
                session_id
            )
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    print(
        "CEMS Engagement Analytics"
    )

    print(
        get_analytics()
    )