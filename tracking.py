import cv2
import math


class StudentTracker:
    def __init__(self, max_distance=150, max_missing=20):
        self.students = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_missing = max_missing

    def _centre(self, face):
        x, y, w, h = face
        return (x + w // 2, y + h // 2)

    def _distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def update(self, faces):
        # Increase missing counter for existing students
        for student_id in self.students:
            self.students[student_id]["missing"] += 1

        results = []
        used_ids = set()

        for face in faces:
            centre = self._centre(face)

            best_id = None
            best_distance = self.max_distance

            # Find closest existing student
            for student_id, student_data in self.students.items():

                if student_id in used_ids:
                    continue

                distance = self._distance(
                    centre,
                    student_data["centre"]
                )

                if distance < best_distance:
                    best_distance = distance
                    best_id = student_id

            # No existing student matched
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

                self.students[best_id] = {
                    "centre": centre,
                    "missing": 0
                }

            else:
                self.students[best_id]["centre"] = centre
                self.students[best_id]["missing"] = 0

            used_ids.add(best_id)

            results.append({
                "student_id": best_id,
                "face": face
            })

        # Remove students missing for too long
        remove_ids = [
            student_id
            for student_id, data in self.students.items()
            if data["missing"] > self.max_missing
        ]

        for student_id in remove_ids:
            del self.students[student_id]

        return results


# -----------------------------------
# Standalone tracking demonstration
# -----------------------------------

if __name__ == "__main__":

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("Error: Face detector could not be loaded.")
        exit()

    tracker = StudentTracker()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open USB camera.")
        exit()

    print("CEMS Student Tracking started.")
    print("Press Q to exit.")

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        tracked_students = tracker.update(faces)

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
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Students Detected: {len(tracked_students)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "CEMS - Student Tracking",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()