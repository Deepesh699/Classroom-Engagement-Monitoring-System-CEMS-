import cv2
import os
import time

from tracking import StudentTracker
from data_processing import save_engagement
from alerts import check_alert
from engagement_service import analyse_engagement


# --------------------------------------------------
# CEMS - Integrated Iteration 1
# --------------------------------------------------

MODEL_PATH = "face_detection_yunet_2023mar.onnx"

SAVE_INTERVAL = 5


def main():

    # --------------------------------------------------
    # CHECK YUNET MODEL
    # --------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        print("ERROR: YuNet model not found:")
        print(MODEL_PATH)
        return


    # --------------------------------------------------
    # YUNET FACE DETECTOR
    # Same detector configuration used in tracking.py
    # --------------------------------------------------

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (320, 320),
        0.75,
        0.3,
        5000
    )


    # --------------------------------------------------
    # STUDENT TRACKER
    # Uses Uday's StudentTracker from tracking.py
    # --------------------------------------------------

    tracker = StudentTracker(
        max_distance=140,
        max_missing=45
    )


    # --------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return


    # Same camera resolution used in Uday's tracking.py

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    print("CEMS Integrated Prototype started.")
    print("YuNet Face Detection: Active")
    print("Student Tracking: Active")
    print("Engagement Analysis: Active")
    print("Data Storage: Active")
    print("Alerts: Active")
    print("Press Q to exit.")


    last_save_time = 0


    # --------------------------------------------------
    # MAIN CAMERA LOOP
    # --------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break


        height, width = frame.shape[:2]


        # --------------------------------------------------
        # 1. YUNET FACE DETECTION
        # --------------------------------------------------

        detector.setInputSize(
            (width, height)
        )

        _, detections = detector.detect(frame)

        faces = []


        if detections is not None:

            for detection in detections:

                x, y, w, h = detection[:4]

                confidence = detection[14]


                # Only accept confident detections

                if confidence >= 0.75:

                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)


                    # Ignore extremely small detections

                    if w >= 45 and h >= 45:

                        faces.append(
                            (x, y, w, h)
                        )


        # --------------------------------------------------
        # 2. STUDENT TRACKING
        # --------------------------------------------------

        tracked_students = tracker.update(
            faces
        )


        # --------------------------------------------------
        # DETERMINE WHETHER DATA SHOULD BE SAVED
        # --------------------------------------------------

        current_time = time.time()

        save_now = (
            current_time - last_save_time
            >= SAVE_INTERVAL
        )


        # Position for alert messages

        alert_y = 100


        # --------------------------------------------------
        # 3. PROCESS EACH TRACKED STUDENT
        # --------------------------------------------------

        for student in tracked_students:

            student_id = student[
                "student_id"
            ]

            face = student[
                "face"
            ]

            x, y, w, h = face


            # --------------------------------------------------
            # 4. ENGAGEMENT ANALYSIS
            # --------------------------------------------------

            result = analyse_engagement(
                width,
                face
            )

            status = result[
                "status"
            ]

            score = result[
                "score"
            ]


            # --------------------------------------------------
            # DRAW STUDENT FACE BOX
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # --------------------------------------------------
            # DISPLAY STUDENT ID + ENGAGEMENT
            # --------------------------------------------------

            label = (
                f"Student {student_id} | "
                f"{status} | "
                f"{score}%"
            )


            cv2.putText(
                frame,
                label,
                (
                    x,
                    max(25, y - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


            # --------------------------------------------------
            # 5. LOW ENGAGEMENT ALERT
            # --------------------------------------------------

            alert_result = check_alert(
                student_id,
                score
            )


            if alert_result["alert"]:

                cv2.putText(
                    frame,
                    alert_result["message"],
                    (20, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                alert_y += 30


            # --------------------------------------------------
            # 6. SAVE ENGAGEMENT DATA
            # Every 5 seconds instead of every frame
            # --------------------------------------------------

            if save_now:

                save_engagement(
                    student_id,
                    score,
                    status
                )


        # Update timer only if there were students

        if save_now and tracked_students:

            last_save_time = current_time


        # --------------------------------------------------
        # 7. DISPLAY STUDENT COUNT
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (340, 65),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            (
                "Students Detected: "
                f"{len(tracked_students)}"
            ),
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # --------------------------------------------------
        # SHOW CEMS WINDOW
        # --------------------------------------------------

        cv2.imshow(
            "CEMS - Integrated Iteration 1",
            frame
        )


        # Q = Exit

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()

    print("CEMS Integrated Prototype stopped.")


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    main()