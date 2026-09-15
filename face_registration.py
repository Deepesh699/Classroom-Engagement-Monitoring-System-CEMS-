import cv2
import numpy as np

from database import get_students
from face_recognition_service import FaceRecognitionService


YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"


def main():
    students = get_students()

    if not students:
        print("No registered students found.")
        return

    print("\nRegistered Students:")
    for student in students:
        print(student)

    student_id = int(
        input("\nEnter the database student ID to enrol: ")
    )

    selected_student = None

    for student in students:
        if student[0] == student_id:
            selected_student = student
            break

    if selected_student is None:
        print("Student not found.")
        return

    student_number = selected_student[1]
    student_name = selected_student[2]

    print(
        f"\nEnrolling face for: "
        f"{student_name} ({student_number})"
    )

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    recognizer = FaceRecognitionService()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open camera.")
        return

    features = []

    print("\nInstructions:")
    print("Look forward at the camera.")
    print("Slowly turn slightly left and right.")
    print("Press SPACE to capture a sample.")
    print("Capture at least 5 samples.")
    print("Press S to save.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        height, width = frame.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        _, detections = detector.detect(
            frame
        )

        current_detection = None

        if detections is not None:
            best_detection = max(
                detections,
                key=lambda item: item[14]
            )

            x, y, w, h = map(
                int,
                best_detection[:4]
            )

            current_detection = best_detection

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Samples: {len(features)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SPACE=capture | S=save | Q=quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "CEMS Face Registration",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if current_detection is None:
                print("No face detected.")
                continue

            feature = recognizer.extract_feature(
                frame,
                current_detection
            )

            features.append(
                feature
            )

            print(
                f"Captured sample {len(features)}"
            )

        elif key == ord("s"):
            if len(features) < 5:
                print(
                    "Please capture at least 5 samples."
                )
                continue

            average_feature = np.mean(
                np.vstack(features),
                axis=0,
                keepdims=True
            )

            recognizer.register_student(
                student_id=student_id,
                student_number=student_number,
                student_name=student_name,
                feature=average_feature
            )

            print(
                f"\nFace enrolled successfully for "
                f"{student_name} ({student_number})"
            )

            break

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()