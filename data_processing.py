import csv
import os
from datetime import datetime

# Project data folder
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
    """Save one engagement result to the CSV file."""
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


def get_engagement_summary():
    """Calculate basic engagement statistics."""
    records = load_engagement_data()

    if not records:
        return {
            "total_students": 0,
            "average_score": 0,
            "engaged": 0,
            "neutral": 0,
            "disengaged": 0
        }

    scores = []

    engaged = 0
    neutral = 0
    disengaged = 0

    for record in records:
        try:
            score = float(record["engagement_score"])
            scores.append(score)
        except (ValueError, TypeError):
            continue

        status = record["status"].lower()

        if status == "engaged":
            engaged += 1
        elif status == "neutral":
            neutral += 1
        elif status == "disengaged":
            disengaged += 1

    average_score = sum(scores) / len(scores) if scores else 0

    return {
        "total_students": len(records),
        "average_score": round(average_score, 2),
        "engaged": engaged,
        "neutral": neutral,
        "disengaged": disengaged
    }


# Test the data processing module
if __name__ == "__main__":

    print("Initialising CEMS data storage...")
    initialise_storage()

    # Test engagement records
    save_engagement(1, 80, "Engaged")
    save_engagement(2, 50, "Neutral")
    save_engagement(3, 25, "Disengaged")

    print("\nTest engagement data saved.")

    print("\nStored Engagement Records:")
    for record in load_engagement_data():
        print(record)

    print("\nEngagement Summary:")
    summary = get_engagement_summary()

    for key, value in summary.items():
        print(f"{key}: {value}") 
        
