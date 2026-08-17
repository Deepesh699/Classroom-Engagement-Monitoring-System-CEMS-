import csv
import os
from datetime import datetime

DATA_FOLDER = "data"
DATA_FILE = os.path.join(DATA_FOLDER, "engagement.csv")


def initialise_storage():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "student_id",
                "engagement_score",
                "status"
            ])


def store_engagement(student_id, engagement_score, status):
    initialise_storage()
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            student_id,
            engagement_score,
            status
        ])


def get_all_records():
    initialise_storage()
    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(row)
    return records


if __name__ == "__main__":
    print("Testing CEMS Data Storage")
    store_engagement(
        student_id=1,
        engagement_score=82,
        status="Engaged"
    )
    store_engagement(
        student_id=2,
        engagement_score=48,
        status="Disengaged"
    )
    print("Records saved successfully.")
    for record in get_all_records():
        print(record)