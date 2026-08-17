import csv
import os
from datetime import datetime

DATA_FOLDER = "data"
DATA_FILE = os.path.join(DATA_FOLDER, "engagement.csv")


def initialise_storage():
    """Create the data folder and CSV file if they do not exist."""
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


def save_engagement(student_id, score, status):
    """Save one engagement result."""
    initialise_storage()

    with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            student_id,
            score,
            status
        ])


def load_engagement_data():
    """Load all stored engagement records."""
    initialise_storage()

    records = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            records.append(row)

    return records


if __name__ == "__main__":
    # Test data for Iteration 1
    save_engagement(1, 80, "Engaged")
    save_engagement(2, 50, "Neutral")
    save_engagement(3, 25, "Disengaged")

    print("Test engagement data saved.")

    for record in load_engagement_data():
        print(record)