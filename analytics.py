from database import (
    get_all_records,
    get_average_engagement,
    get_low_engagement_count
)


def get_analytics():

    records = get_all_records()

    return {
        "average_engagement":
            get_average_engagement(),

        "records":
            len(records),

        "low_engagement_records":
            get_low_engagement_count(60)
    }


if __name__ == "__main__":

    analytics = get_analytics()

    print("CEMS Analytics")
    print("-------------------------")

    print(
        "Average Engagement:",
        analytics["average_engagement"],
        "%"
    )

    print(
        "Total Records:",
        analytics["records"]
    )

    print(
        "Low Engagement Records:",
        analytics["low_engagement_records"]
    )