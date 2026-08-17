import cv2
import math
import os


MODEL_PATH = "face_detection_yunet_2023mar.onnx"


class StudentTracker:
    def __init__(self, max_distance=140, max_missing=45):
        self.students = {}
        self.max_distance = max_distance
        self.max_missing = max_missing

    def centre(self, box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)

        width = max(0, right - left)
        height = max(0, bottom - top)

        intersection = width * height

        area1 = w1 * h1
        area2 = w2 * h2

        union = area1 + area2 - intersection

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

    def update_student(self, student_id, face):
        student = self.students[student_id]

        new_centre = self.centre(face)
        old_centre = student["centre"]

        velocity_x = new_centre[0] - old_centre[0]
        velocity_y = new_centre[1] - old_centre[1]

        student["velocity"] = (
            student["velocity"][0] * 0.7 + velocity_x * 0.3,
            student["velocity"][1] * 0.7 + velocity_y * 0.3
        )

        student["centre"] = new_centre
        student["box"] = face
        student["missing"] = 0

    def create_student(self, face):
        student_id = self.next_id()

        centre = self.centre(face)

        self.students[student_id] = {
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

        for student in self.students.values():
            student["missing"] += 1

        if not faces:
            self.remove_missing()

            return []

        matches = []
        used_students = set()
        used_faces = set()

        candidates = []

        for student_id, student in self.students.items():
            predicted = self.predict(student)

            for face_index, face in enumerate(faces):
                face_centre = self.centre(face)

                distance = self.distance(
                    predicted,
                    face_centre
                )

                overlap = self.iou(
                    student["box"],
                    face
                )

                if distance <= self.max_distance or overlap > 0.05:
                    distance_score = max(
                        0,
                        1 - distance / self.max_distance
                    )

                    score = (
                        distance_score * 0.65 +
                        overlap * 0.35
                    )

                    candidates.append(
                        (
                            score,
                            student_id,
                            face_index
                        )
                    )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        for score, student_id, face_index in candidates:

            if student_id in used_students:
                continue

            if face_index in used_faces:
                continue

            used_students.add(student_id)
            used_faces.add(face_index)

            matches.append(
                (
                    student_id,
                    face_index
                )
            )

        for student_id, face_index in matches:
            self.update_student(
                student_id,
                faces[face_index]
            )

        for face_index, face in enumerate(faces):

            if face_index not in used_faces:
                self.create_student(face)

        self.remove_missing()

        results = []

        for student_id, student in self.students.items():

            if student["missing"] == 0:
                results.append({
                    "student_id": student_id,
                    "face": student["box"]
                })

        results.sort(
            key=lambda x: x["student_id"]
        )

        return results

    def remove_missing(self):
        remove_ids = [
            student_id
            for student_id, student in self.students.items()
            if student["missing"] > self.max_missing
        ]

        for student_id in remove_ids:
            del self.students[student_id]


def main():

    if not os.path.exists(MODEL_PATH):
        print(
            "ERROR: YuNet model not found:"
        )
        print(MODEL_PATH)
        return

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (320, 320),
        0.75,
        0.3,
        5000
    )

    tracker = StudentTracker(
        max_distance=140,
        max_missing=45
    )

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

    print("CEMS Student Tracking started.")
    print("Press Q to exit.")

    while True:

        success, frame = camera.read()

        if not success:
            break

        height, width = frame.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        _, detections = detector.detect(frame)

        faces = []

        if detections is not None:

            for detection in detections:

                x, y, w, h = detection[:4]
                confidence = detection[14]

                if confidence >= 0.75:

                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)

                    if w >= 45 and h >= 45:

                        faces.append(
                            (x, y, w, h)
                        )

        tracked_students = tracker.update(
            faces
        )

        for student in tracked_students:

            student_id = student["student_id"]

            x, y, w, h = student["face"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Student {student_id}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.rectangle(
            frame,
            (10, 10),
            (310, 65),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Students Detected: {len(tracked_students)}",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "CEMS - Student Tracking",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()

    cv2.destroyAllWindows()

    print("CEMS Student Tracking stopped.")


if __name__ == "__main__":
    main()