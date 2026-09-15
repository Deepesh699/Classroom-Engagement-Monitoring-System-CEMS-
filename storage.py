import os
import sqlite3
from datetime import datetime

DATA_FOLDER = "data"
DATABASE_FILE = os.path.join(DATA_FOLDER, "cems.db")


def get_connection():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row

    return connection


def get_all_records():
    connection = get_connection()

    try:
        cursor = connection.execute("""
            SELECT *
            FROM engagement_records
            ORDER BY timestamp DESC
        """)

        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def store_engagement(
    student_id,
    engagement_score,
    status,
    registered_student_id=None,
    session_id=None
):
    connection = get_connection()

    try:
        connection.execute("""
            INSERT INTO engagement_records
            (
                timestamp,
                student_id,
                engagement_score,
                status,
                registered_student_id,
                session_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            student_id,
            engagement_score,
            status,
            registered_student_id,
            session_id
        ))

        connection.commit()

    finally:
        connection.close()


if __name__ == "__main__":
    print("Testing SQLite engagement storage")

    records = get_all_records()

    print(f"Records found: {len(records)}")

    for record in records[:10]:
        print(record)