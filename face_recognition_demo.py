import cv2

from face_recognition_service import FaceRecognitionService


YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"


def main():

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    recognition_service = FaceRecognitionService(
        threshold=0.40
    )

    print(
        f"Loaded {len(recognition_service.registered_faces)} "
        f"registered student(s)."
    )

    if not recognition_service.registered_faces:
        print(
            "No enrolled faces found. "
            "Run face_registration.py first."
        )
        return

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

    print("\nCEMS Face Recognition started.")
    print("Press Q to exit.")

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

        if detections is not None:

            for detection in detections:

                confidence = float(
                    detection[14]
                )

                if confidence < 0.8:
                    continue

                x, y, w, h = map(
                    int,
                    detection[:4]
                )

                try:

                    feature = (
                        recognition_service.extract_feature(
                            frame,
                            detection
                        )
                    )

                    result = (
                        recognition_service.recognise(
                            feature
                        )
                    )

                    similarity = result[
                        "similarity"
                    ]

                    if result["recognised"]:

                        student_name = result[
                            "student_name"
                        ]

                        student_number = result[
                            "student_number"
                        ]

                        label = (
                            f"{student_name} | "
                            f"{student_number} | "
                            f"{similarity:.2f}"
                        )

                        box_color = (
                            0,
                            255,
                            0
                        )

                    else:

                        label = (
                            f"Unknown | "
                            f"{similarity:.2f}"
                        )

                        box_color = (
                            0,
                            0,
                            255
                        )

                except cv2.error:

                    label = (
                        "Recognition error"
                    )

                    box_color = (
                        0,
                        165,
                        255
                    )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    2
                )

                text_y = max(
                    30,
                    y - 10
                )

                cv2.putText(
                    frame,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2
                )

        cv2.putText(
            frame,
            "CEMS Registered Student Recognition",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = Quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "CEMS Face Recognition",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()