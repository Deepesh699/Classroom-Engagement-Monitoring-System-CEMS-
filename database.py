import sqlite3
import os
from datetime import datetime

DATA_FOLDER = "data"
DB_PATH = os.path.join(DATA_FOLDER, "cems.db")


def get_connection():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# -------------------------------------------------
# DATABASE INITIALISATION
# -------------------------------------------------

def initialise_database():
    connection = get_connection()
    cursor = connection.cursor()

    # Registered students
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_number TEXT NOT NULL UNIQUE,
            student_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Units / subjects
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_code TEXT NOT NULL UNIQUE,
            unit_name TEXT NOT NULL
        )
    """)

    # Classrooms
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classrooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classroom_name TEXT NOT NULL UNIQUE
        )
    """)

    # Classroom sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id INTEGER NOT NULL,
            classroom_id INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            FOREIGN KEY (unit_id) REFERENCES units(id),
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id)
        )
    """)

    # Keep the existing engagement table
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

    # Add new Iteration 2 columns to the existing table
    cursor.execute("PRAGMA table_info(engagement_records)")
    columns = [column[1] for column in cursor.fetchall()]

    if "registered_student_id" not in columns:
        cursor.execute("""
            ALTER TABLE engagement_records
            ADD COLUMN registered_student_id INTEGER
        """)

    if "session_id" not in columns:
        cursor.execute("""
            ALTER TABLE engagement_records
            ADD COLUMN session_id INTEGER
        """)

    connection.commit()
    connection.close()


# -------------------------------------------------
# STUDENTS
# -------------------------------------------------

def add_student(student_number, student_name):
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO students (
                student_number,
                student_name,
                created_at
            )
            VALUES (?, ?, ?)
        """, (
            student_number,
            student_name,
            datetime.now().isoformat(timespec="seconds")
        ))

        connection.commit()

        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()


def get_students():
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            student_number,
            student_name,
            created_at
        FROM students
        ORDER BY student_name ASC
    """)

    students = cursor.fetchall()

    connection.close()

    return students


# -------------------------------------------------
# UNITS
# -------------------------------------------------

def add_unit(unit_code, unit_name):
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO units (
                unit_code,
                unit_name
            )
            VALUES (?, ?)
        """, (
            unit_code,
            unit_name
        ))

        connection.commit()

        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()


def get_units():
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            unit_code,
            unit_name
        FROM units
        ORDER BY unit_code ASC
    """)

    units = cursor.fetchall()

    connection.close()

    return units


# -------------------------------------------------
# CLASSROOMS
# -------------------------------------------------

def add_classroom(classroom_name):
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO classrooms (
                classroom_name
            )
            VALUES (?)
        """, (classroom_name,))

        connection.commit()

        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()


def get_classrooms():
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            classroom_name
        FROM classrooms
        ORDER BY classroom_name ASC
    """)

    classrooms = cursor.fetchall()

    connection.close()

    return classrooms


# -------------------------------------------------
# CLASS SESSIONS
# -------------------------------------------------

def start_session(unit_id, classroom_id):
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO sessions (
            unit_id,
            classroom_id,
            start_time
        )
        VALUES (?, ?, ?)
    """, (
        unit_id,
        classroom_id,
        datetime.now().isoformat(timespec="seconds")
    ))

    session_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return session_id


def end_session(session_id):
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE sessions
        SET end_time = ?
        WHERE id = ?
    """, (
        datetime.now().isoformat(timespec="seconds"),
        session_id
    ))

    connection.commit()
    connection.close()


def get_sessions():
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            sessions.id,
            units.unit_code,
            units.unit_name,
            classrooms.classroom_name,
            sessions.start_time,
            sessions.end_time
        FROM sessions
        JOIN units
            ON sessions.unit_id = units.id
        JOIN classrooms
            ON sessions.classroom_id = classrooms.id
        ORDER BY sessions.id DESC
    """)

    sessions = cursor.fetchall()

    connection.close()

    return sessions


# -------------------------------------------------
# ENGAGEMENT RECORDS
# -------------------------------------------------

def save_engagement(
    student_id,
    engagement_score,
    status,
    registered_student_id=None,
    session_id=None
):
    initialise_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO engagement_records (
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
            status,
            registered_student_id,
            session_id
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


# -------------------------------------------------
# TEST
# -------------------------------------------------

if __name__ == "__main__":
    initialise_database()

    print("CEMS database created successfully.")
    print("Database location:", DB_PATH)

    print("\nDatabase tables ready:")
    print("- students")
    print("- units")
    print("- classrooms")
    print("- sessions")
    print("- engagement_records")
    