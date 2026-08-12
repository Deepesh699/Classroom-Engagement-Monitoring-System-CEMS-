import cv2
import time

from camera import open_camera, release_camera
from detection import FaceDetector, draw_faces
from engagement import EngagementAnalyzer
from tracking import StudentTracker
from data_processing import save_engagement
from alerts import check_alert


camera = open_camera()

detector = FaceDetector()
analyzer = EngagementAnalyzer()
tracker = StudentTracker()

last_save_time = 0

print("CEMS Iteration 1 Prototype Running")
print("Press Q to exit.")


while True:

    success, frame = camera.read()

    if not success:
        break

    faces = detector.detect_faces(frame)

    tracked_students = tracker.update(faces)

    frame_width = frame.shape[1]

    y_position = 80

    for student in tracked_students:

        student_id = student["student_id"]
        face = student["face"]

        result = analyzer.analyse(
            frame_width,
            face
        )

        score = result["score"]
        status = result["status"]

        alert = check_alert(
            student_id,
            score
        )

        display_text = (
            f"Student {student_id}: "
            f"{status} {score}%"
        )

        cv2.putText(
            frame,
            display_text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y_position += 30

        if alert["alert"]:

            cv2.putText(
                frame,
                alert["message"],
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

            y_position += 30

        # Store data every 5 seconds
        if time.time() - last_save_time >= 5:

            save_engagement(
                student_id,
                score,
                status
            )

    if time.time() - last_save_time >= 5:
        last_save_time = time.time()

    frame = draw_faces(
        frame,
        faces
    )

    cv2.imshow(
        "CEMS Iteration 1",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


release_camera(camera)