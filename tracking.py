import math


class StudentTracker:

    def __init__(self, max_distance=120):
        self.students = {}
        self.next_id = 1
        self.max_distance = max_distance

    def _centre(self, face):
        x, y, w, h = face
        return (x + w // 2, y + h // 2)

    def _distance(self, point1, point2):
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2
        )

    def update(self, faces):
        updated_students = {}
        results = []

        for face in faces:
            centre = self._centre(face)

            matched_id = None
            smallest_distance = self.max_distance

            for student_id, previous_centre in self.students.items():
                distance = self._distance(
                    centre,
                    previous_centre
                )

                if distance < smallest_distance:
                    smallest_distance = distance
                    matched_id = student_id

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            updated_students[matched_id] = centre

            results.append({
                "student_id": matched_id,
                "face": face
            })

        self.students = updated_students
        return results