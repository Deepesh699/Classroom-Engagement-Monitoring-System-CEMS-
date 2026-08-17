import csv
import os
DATA_FILE = os.path.join(
    "data",
    "engagement.csv"
)
def get_analytics():
    if not os.path.exists(DATA_FILE):
        return {
            "average_engagement": 0,
            "records": 0,
            "low_engagement_records": 0
        }
    scores = []
    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            scores.append(
                int(row["engagement_score"])
            )
    if not scores:
        average = 0
    else:
        average = sum(scores) / len(scores)
    low_count = len([
        score
        for score in scores
        if score < 60
    ])
    return {
        "average_engagement": round(
            average,
            2
        ),
        "records": len(scores),
        "low_engagement_records": low_count
    }
if __name__ == "__main__":
    print(get_analytics())