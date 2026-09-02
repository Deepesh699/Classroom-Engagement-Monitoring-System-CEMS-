import cv2
import math
import os

from engagement_service import analyse_engagement


MODEL_PATH = "face_detection_yunet_2023mar.onnx"


class StudentTracker:

    def __init__(self, max_distance=140, max_missing=45):
        self.students = {}
        self.max_distance = max_distance
        self.max_missing = max_missing

    def centre(self, box):
        x, y, w, h = box

        return (
            x + w / 2,
            y + h / 2
        )

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2
            +
            (p1[1] - p2[1]) ** 2
        )

    def iou(self, box1, box2):

        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        left = max(x1, x2)
        top = max(y1, y2)

        right = min(
            x1 + w1,
            x2 + w2
        )

        bottom = min(
            y1 + h1,
            y2 + h2
        )

        width = max(
            0,
            right - left
        )

        height = max(
            0,
            bottom - top
        )

        intersection = width * height

        area1 = w1 * h1
        area2 = w2 * h2

        union = (
            area1
            + area2
            - intersection
        )

        if union <= 0:
            return 0

        return intersection / union

    def next_id(self):

        student_id = 1

        while student_id in self.students:
            student_id += 1

        return student_id

    def predict(self, student):

        cx, cy = student["centre"]
        vx, vy = student["velocity"]

        return (
            cx + vx,
            cy + vy
        )

    def update_student(
        self,
        student_id,
        face
    ):

        student = self.students[
            student_id
        ]

        new_centre = self.centre(
            face
        )

        old_centre = student[
            "centre"
        ]

        velocity_x = (
            new_centre[0]
            - old_centre[0]
        )

        velocity_y = (
            new_centre[1]
            - old_centre[1]
        )

        student["velocity"] = (
            student["velocity"][0]
            * 0.7
            + velocity_x
            * 0.3,

            student["velocity"][1]
            * 0.7
            + velocity_y
            * 0.3
        )

        student["centre"] = new_centre
        student["box"] = face
        student["missing"] = 0

    def create_student(
        self,
        face
    ):

        student_id = self.next_id()

        centre = self.centre(
            face
        )

        self.students[
            student_id
        ] = {
            "box": face,
            "centre": centre,
            "velocity": (0, 0),
            "missing": 0
        }

        return student_id

    def update(self, faces):

        faces = [
            tuple(map(int, face))
            for face in faces
        ]

        # Mark students as missing until
        # they are matched in this frame.
        for student in self.students.values():
            student["missing"] += 1

        if not faces:

            self.remove_missing()

            return []

        matches = []

        used_students = set()
        used_faces = set()

        candidates = []

        # --------------------------------
        # CREATE MATCHING CANDIDATES
        # --------------------------------

        for (
            student_id,
            student
        ) in self.students.items():

            predicted = self.predict(
                student
            )

            for (
                face_index,
                face
            ) in enumerate(faces):

                face_centre = self.centre(
                    face
                )

                distance = self.distance(
                    predicted,
                    face_centre
                )

                overlap = self.iou(
                    student["box"],
                    face
                )

                if (
                    distance <= self.max_distance
                    or overlap > 0.05
                ):

                    distance_score = max(
                        0,
                        1
                        - distance
                        / self.max_distance
                    )

                    score = (
                        distance_score
                        * 0.65
                        +
                        overlap
                        * 0.35
                    )

                    candidates.append(
                        (
                            score,
                            student_id,
                            face_index
                        )
                    )

        # Best matches first.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        # --------------------------------
        # MATCH FACES WITH STUDENTS
        # --------------------------------

        for (
            score,
            student_id,
            face_index
        ) in candidates:

            if student_id in used_students:
                continue

            if face_index in used_faces:
                continue

            used_students.add(
                student_id
            )

            used_faces.add(
                face_index
            )

            matches.append(
                (
                    student_id,
                    face_index
                )
            )

        # Update matched students.
        for (
            student_id,
            face_index
        ) in matches:

            self.update_student(
                student_id,
                faces[face_index]
            )

        # Create IDs for new students.
        for (
            face_index,
            face
        ) in enumerate(faces):

            if face_index not in used_faces:

                self.create_student(
                    face
                )

        self.remove_missing()

        results = []

        for (
            student_id,
            student
        ) in self.students.items():

            if student["missing"] == 0:

                results.append(
                    {
                        "student_id": student_id,
                        "face": student["box"]
                    }
                )

        results.sort(
            key=lambda x: x["student_id"]
        )

        return results

    def remove_missing(self):

        remove_ids = [

            student_id

            for (
                student_id,
                student
            ) in self.students.items()

            if student["missing"]
            > self.max_missing
        ]

        for student_id in remove_ids:

            del self.students[
                student_id
            ]


def find_best_detection(
    student_face,
    faces,
    detections
):

    """
    Connect a tracked student ID with
    the closest current YuNet detection.
    """

    sx, sy, sw, sh = student_face

    student_centre = (
        sx + sw / 2,
        sy + sh / 2
    )

    best_detection = None
    best_distance = float("inf")

    for index, face in enumerate(faces):

        fx, fy, fw, fh = face

        face_centre = (
            fx + fw / 2,
            fy + fh / 2
        )

        distance = math.sqrt(
            (
                student_centre[0]
                - face_centre[0]
            ) ** 2
            +
            (
                student_centre[1]
                - face_centre[1]
            ) ** 2
        )

        if distance < best_distance:

            best_distance = distance

            if index < len(detections):

                best_detection = (
                    detections[index]
                )

    return best_detection


def main():

    # --------------------------------
    # CHECK YUNET MODEL
    # --------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "ERROR: YuNet model not found:"
        )

        print(MODEL_PATH)

        return

    # --------------------------------
    # CREATE YUNET FACE DETECTOR
    # --------------------------------

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (320, 320),
        0.75,
        0.3,
        5000
    )

    # --------------------------------
    # CREATE TRACKER
    # --------------------------------

    tracker = StudentTracker(
        max_distance=140,
        max_missing=45
    )

    # --------------------------------
    # OPEN CAMERA
    # --------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print(
        "CEMS Engagement Monitoring started."
    )

    print(
        "Press Q to exit."
    )

    # --------------------------------
    # MAIN LOOP
    # --------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Could not read camera frame."
            )

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

        # --------------------------------
        # PROCESS YUNET RESULTS
        # --------------------------------

        if detections is not None:

            for detection in detections:

                x, y, w, h = (
                    detection[:4]
                )

                confidence = (
                    detection[14]
                )

                if confidence >= 0.75:

                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)

                    if (
                        w >= 45
                        and h >= 45
                    ):

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

        # --------------------------------
        # TRACK STUDENTS
        # --------------------------------

        tracked_students = (
            tracker.update(faces)
        )

        engaged_count = 0
        neutral_count = 0
        low_count = 0
        live_results = []

        # --------------------------------
        # PROCESS EACH STUDENT
        # --------------------------------

        for student in tracked_students:

            student_id = (
                student["student_id"]
            )

            face = student["face"]

            x, y, w, h = face

            detection = find_best_detection(
                face,
                faces,
                valid_detections
            )

            # --------------------------------
            # ENGAGEMENT ANALYSIS
            # --------------------------------

            if detection is not None:

                result = analyse_engagement(
                    detection
                )

                status = result[
                    "status"
                ]

                score = result[
                    "score"
                ]

                orientation = result[
                    "orientation"
                ]

            else:

                status = "Unknown"
                score = 0
                orientation = "Unknown"
                live_results.append({
    "track_id": student_id,
    "score": score,
    "status": status,
    "orientation": orientation
})

            # --------------------------------
            # CLASSROOM COUNTS
            # --------------------------------

            if status == "Engaged":

                engaged_count += 1

            elif status == "Neutral":

                neutral_count += 1

            elif status == "Low Engagement":

                low_count += 1

            # --------------------------------
            # BOX COLOUR
            # --------------------------------

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

            # --------------------------------
            # DRAW FACE BOX
            # --------------------------------

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

            # --------------------------------
            # STUDENT LABEL
            # --------------------------------

            label = (
                f"Student {student_id} | "
                f"{orientation} | "
                f"{status} | "
                f"{score}%"
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
                0.52,
                box_colour,
                2,
                cv2.LINE_AA
            )

        # --------------------------------
        # CLASSROOM SUMMARY BOX
        # --------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (420, 90),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            (
                "Students Detected: "
                f"{len(tracked_students)}"
            ),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "Engaged: "
                f"{engaged_count}"
            ),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "Neutral: "
                f"{neutral_count}"
            ),
            (150, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "Low: "
                f"{low_count}"
            ),
            (285, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # --------------------------------
        # DISPLAY CAMERA
        # --------------------------------

        cv2.imshow(
            "CEMS - Engagement Monitoring",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    # --------------------------------
    # CLEANUP
    # --------------------------------

    camera.release()

    cv2.destroyAllWindows()

    print(
        "CEMS session ended."
    )


if __name__ == "__main__":
    main()