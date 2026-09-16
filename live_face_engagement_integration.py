import time
from collections import defaultdict, deque

import cv2

from tracking import StudentTracker, find_best_detection
from engagement_service import analyse_engagement
from face_recognition_service import FaceRecognitionService

from database import (
    get_active_session_id,
    get_student_by_id,
    get_registered_student_for_track,
    assign_track_to_student,
    save_live_results
)


YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

SAVE_INTERVAL_SECONDS = 5
RECOGNITION_CONFIRM_FRAMES = 3


def main():

    # -------------------------------------------------
    # ACTIVE SESSION
    # -------------------------------------------------

    session_id = get_active_session_id()

    if session_id is None:
        print("ERROR: No active classroom session.")
        print("Start a session before running live integration.")
        return

    print(f"Active session: {session_id}")

    # -------------------------------------------------
    # FACE DETECTOR
    # -------------------------------------------------

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        0.75,
        0.3,
        5000
    )

    # -------------------------------------------------
    # TRACKER
    # -------------------------------------------------

    tracker = StudentTracker(
        max_distance=140,
        max_missing=45
    )

    # -------------------------------------------------
    # FACE RECOGNITION
    # -------------------------------------------------

    face_recognition = FaceRecognitionService(
        threshold=0.40
    )

    print(
        "Registered face profiles:",
        len(face_recognition.registered_faces)
    )

    if not face_recognition.registered_faces:
        print("ERROR: No enrolled students.")
        print("Run face_registration.py first.")
        return

    # -------------------------------------------------
    # IDENTITY MEMORY
    # -------------------------------------------------

    # Once a track has been confidently recognised,
    # keep that student identity stable for the session.
    track_identity = {}

    # Used to prevent one random recognition frame
    # from assigning the wrong student.
    recognition_history = defaultdict(
        lambda: deque(
            maxlen=RECOGNITION_CONFIRM_FRAMES
        )
    )

    # -------------------------------------------------
    # CAMERA
    # -------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    last_save_time = time.time()

    print()
    print("CEMS LIVE INTEGRATION STARTED")
    print("Tracking + Face Recognition + Engagement + SQLite")
    print("Press Q to exit.")
    print()

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        height, width = frame.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        _, detections = detector.detect(
            frame
        )

        faces = []
        valid_detections = []

        # -------------------------------------------------
        # YUNET DETECTION
        # -------------------------------------------------

        if detections is not None:

            for detection in detections:

                confidence = float(
                    detection[14]
                )

                if confidence < 0.75:
                    continue

                x, y, w, h = map(
                    int,
                    detection[:4]
                )

                if w < 45 or h < 45:
                    continue

                faces.append(
                    (
                        x,
                        y,
                        w,
                        h
                    )
                )

                valid_detections.append(
                    detection
                )

        # -------------------------------------------------
        # TRACKING
        # -------------------------------------------------

        tracked_students = tracker.update(
            faces
        )

        live_results = []

        engaged_count = 0
        neutral_count = 0
        low_count = 0

        # -------------------------------------------------
        # EACH TRACK
        # -------------------------------------------------

        for tracked_student in tracked_students:

            track_id = tracked_student[
                "track_id"
            ]

            face_box = tracked_student[
                "face"
            ]

            x, y, w, h = face_box

            detection = find_best_detection(
                face_box,
                faces,
                valid_detections
            )

            # -------------------------------------------------
            # ENGAGEMENT
            # -------------------------------------------------

            if detection is not None:

                engagement = analyse_engagement(
                    detection
                )

                score = engagement[
                    "score"
                ]

                status = engagement[
                    "status"
                ]

                orientation = engagement[
                    "orientation"
                ]

            else:

                score = 0
                status = "Unknown"
                orientation = "Unknown"

            # -------------------------------------------------
            # CHECK EXISTING TRACK ASSIGNMENT
            # -------------------------------------------------

            if track_id not in track_identity:

                existing_student_id = (
                    get_registered_student_for_track(
                        session_id,
                        track_id
                    )
                )

                if existing_student_id is not None:

                    track_identity[
                        track_id
                    ] = existing_student_id

            # -------------------------------------------------
            # FACE RECOGNITION
            # -------------------------------------------------

            recognition_similarity = 0.0

            if (
                detection is not None
                and track_id not in track_identity
            ):

                try:

                    feature = (
                        face_recognition.extract_feature(
                            frame,
                            detection
                        )
                    )

                    recognition_result = (
                        face_recognition.recognise(
                            feature
                        )
                    )

                    recognition_similarity = (
                        recognition_result[
                            "similarity"
                        ]
                    )

                    if recognition_result[
                        "recognised"
                    ]:

                        recognised_student_id = (
                            recognition_result[
                                "student_id"
                            ]
                        )

                        recognition_history[
                            track_id
                        ].append(
                            recognised_student_id
                        )

                        history = list(
                            recognition_history[
                                track_id
                            ]
                        )

                        # Require several consecutive
                        # identical recognition results.
                        if (
                            len(history)
                            == RECOGNITION_CONFIRM_FRAMES
                            and len(set(history)) == 1
                        ):

                            confirmed_student_id = (
                                history[0]
                            )

                            database_student = (
                                get_student_by_id(
                                    confirmed_student_id
                                )
                            )

                            if database_student is not None:

                                track_identity[
                                    track_id
                                ] = (
                                    confirmed_student_id
                                )

                                assign_track_to_student(
                                    session_id,
                                    track_id,
                                    confirmed_student_id
                                )

                                print(
                                    f"Track {track_id} "
                                    f"recognised as "
                                    f"{database_student[2]} "
                                    f"({database_student[1]})"
                                )

                    else:

                        recognition_history[
                            track_id
                        ].clear()

                except cv2.error as error:

                    print(
                        "Face recognition error:",
                        error
                    )

            # -------------------------------------------------
            # GET REGISTERED STUDENT
            # -------------------------------------------------

            registered_student_id = (
                track_identity.get(
                    track_id
                )
            )

            student_name = None
            student_number = None

            if registered_student_id is not None:

                student = get_student_by_id(
                    registered_student_id
                )

                if student is not None:

                    student_number = student[1]
                    student_name = student[2]

            # -------------------------------------------------
            # CREATE LIVE RESULT
            # -------------------------------------------------

            live_result = {
                "track_id": track_id,
                "score": score,
                "status": status,
                "orientation": orientation
            }

            if registered_student_id is not None:

                live_result[
                    "registered_student_id"
                ] = registered_student_id

            live_results.append(
                live_result
            )

            # -------------------------------------------------
            # COUNTS
            # -------------------------------------------------

            if status == "Engaged":
                engaged_count += 1

            elif status == "Neutral":
                neutral_count += 1

            elif status == "Low Engagement":
                low_count += 1

            # -------------------------------------------------
            # DISPLAY LABEL
            # -------------------------------------------------

            if registered_student_id is not None:

                label = (
                    f"{student_name} | "
                    f"{student_number} | "
                    f"{orientation} | "
                    f"{status} | "
                    f"{score}%"
                )

            else:

                label = (
                    f"Track {track_id} | "
                    f"Unknown Student | "
                    f"{orientation} | "
                    f"{status} | "
                    f"{score}%"
                )

            # -------------------------------------------------
            # BOX COLOR
            # -------------------------------------------------

            if status == "Engaged":

                box_colour = (
                    0,
                    255,
                    0
                )

            elif status == "Neutral":

                box_colour = (
                    0,
                    255,
                    255
                )

            else:

                box_colour = (
                    0,
                    0,
                    255
                )

            cv2.rectangle(
                frame,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                box_colour,
                2
            )

            cv2.putText(
                frame,
                label,
                (
                    x,
                    max(
                        25,
                        y - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                box_colour,
                2,
                cv2.LINE_AA
            )

        # -------------------------------------------------
        # CONTROLLED DATABASE SAVE
        # -------------------------------------------------

        current_time = time.time()

        if (
            live_results
            and current_time
            - last_save_time
            >= SAVE_INTERVAL_SECONDS
        ):

            saved_count = save_live_results(
                session_id,
                live_results
            )

            if saved_count > 0:

                print(
                    f"Saved {saved_count} "
                    f"engagement record(s) "
                    f"for session {session_id}"
                )

            last_save_time = current_time

        # -------------------------------------------------
        # CLASSROOM SUMMARY
        # -------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (500, 95),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Session: {session_id}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"Tracked: "
                f"{len(tracked_students)} | "
                f"Engaged: {engaged_count} | "
                f"Neutral: {neutral_count} | "
                f"Low: {low_count}"
            ),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        cv2.imshow(
            "CEMS - Live Integrated System",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------

    camera.release()
    cv2.destroyAllWindows()

    print("CEMS live session stopped.")


if __name__ == "__main__":
    main()