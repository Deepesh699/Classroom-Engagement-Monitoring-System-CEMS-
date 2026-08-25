import sqlite3
import os
from datetime import datetime

DATA_FOLDER = "data"
DB_PATH = os.path.join(DATA_FOLDER, "cems.db")


def get_connection():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialise_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS engagement_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            student_id INTEGER NOT NULL,
            engagement_score INTEGER NOT NULL,
            status TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_engagement(student_id, engagement_score, status):
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO engagement_records
        (
            timestamp,
            student_id,
            engagement_score,
            status
        )
        VALUES (?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        student_id,
        engagement_score,
        status
    ))

    connection.commit()
    connection.close()


def get_all_records():
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            student_id,
            engagement_score,
            status
        FROM engagement_records
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    connection.close()

    return records


def get_average_engagement():
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT AVG(engagement_score)
        FROM engagement_records
    """)

    result = cursor.fetchone()[0]

    connection.close()

    if result is None:
        return 0

    return round(result, 2)


def get_low_engagement_count(threshold=60):
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM engagement_records
        WHERE engagement_score < ?
    """, (threshold,))

    result = cursor.fetchone()[0]

    connection.close()

    return result


if __name__ == "__main__":
    initialise_database()

    print("CEMS database created successfully.")
    print("Database location:", DB_PATH)