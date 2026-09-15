import os
import pickle

import cv2
import numpy as np


SFACE_MODEL_PATH = os.path.join(
    "models",
    "face_recognition_sface_2021dec.onnx"
)

EMBEDDINGS_PATH = os.path.join(
    "data",
    "face_embeddings",
    "registered_faces.pkl"
)


class FaceRecognitionService:
    def __init__(self, threshold=0.40):
        if not os.path.exists(SFACE_MODEL_PATH):
            raise FileNotFoundError(
                f"SFace model not found: {SFACE_MODEL_PATH}"
            )

        self.recognizer = cv2.FaceRecognizerSF.create(
            SFACE_MODEL_PATH,
            ""
        )

        self.threshold = threshold
        self.registered_faces = {}

        self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, "rb") as file:
                self.registered_faces = pickle.load(file)
        else:
            self.registered_faces = {}

    def save_embeddings(self):
        os.makedirs(
            os.path.dirname(EMBEDDINGS_PATH),
            exist_ok=True
        )

        with open(EMBEDDINGS_PATH, "wb") as file:
            pickle.dump(
                self.registered_faces,
                file
            )

    def extract_feature(self, frame, detection):
        detection = np.asarray(
            detection,
            dtype=np.float32
        )

        aligned_face = self.recognizer.alignCrop(
            frame,
            detection
        )

        feature = self.recognizer.feature(
            aligned_face
        )

        return feature

    def register_student(
        self,
        student_id,
        student_number,
        student_name,
        feature
    ):
        self.registered_faces[int(student_id)] = {
            "student_number": student_number,
            "student_name": student_name,
            "feature": feature
        }

        self.save_embeddings()

    def recognise(self, feature):
        best_student = None
        best_score = -1.0

        for student_id, student in self.registered_faces.items():

            stored_feature = student["feature"]

            similarity = self.recognizer.match(
                stored_feature,
                feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            if similarity > best_score:
                best_score = similarity

                best_student = {
                    "student_id": student_id,
                    "student_number": student["student_number"],
                    "student_name": student["student_name"]
                }

        if (
            best_student is not None
            and best_score >= self.threshold
        ):
            best_student["similarity"] = round(
                float(best_score),
                3
            )

            best_student["recognised"] = True

            return best_student

        return {
            "student_id": None,
            "student_number": None,
            "student_name": "Unknown",
            "similarity": (
                round(float(best_score), 3)
                if best_score >= 0
                else 0.0
            ),
            "recognised": False
        }