import cv2
import math
import os
import time
from collections import Counter, deque

from engagement_service import analyse_engagement


MODEL_PATH = "face_detection_yunet_2023mar.onnx"


# ============================================================
# STUDENT TRACKER
# ============================================================

class StudentTracker:

    def __init__(self, max_distance=140, max_missing=45):
        self.students = {}
        self.max_distance = max_distance
        self.max_missing = max_missing

        # IDs keep increasing instead of being recycled.
        self.next_student_id = 1

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

    def predict(self, student):

        cx, cy = student["centre"]
        vx, vy = student["velocity"]

        return (
            cx + vx,
            cy + vy
        )

    def create_student(self, face):

        student_id = self.next_student_id

        self.next_student_id += 1

        self.students[student_id] = {
            "box": face,
            "centre": self.centre(face),
            "velocity": (0, 0),
            "missing": 0
        }

        return student_id

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

        # Smooth velocity.
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

    def update(self, faces):

        faces = [
            tuple(map(int, face))
            for face in faces
        ]

        # First mark all existing students missing.
        for student in self.students.values():
            student["missing"] += 1

        if not faces:

            self.remove_missing()

            return []

        matches = []

        used_students = set()
        used_faces = set()

        candidates = []

        # ====================================================
        # CREATE MATCHING CANDIDATES
        # ====================================================

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

                    # 65% movement distance
                    # 35% bounding-box overlap
                    match_score = (
                        distance_score
                        * 0.65
                        +
                        overlap
                        * 0.35
                    )

                    candidates.append(
                        (
                            match_score,
                            student_id,
                            face_index
                        )
                    )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        # ====================================================
        # ONE-TO-ONE MATCHING
        # ====================================================

        for (
            match_score,
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

        # New unmatched faces get new IDs.
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
            key=lambda item:
            item["student_id"]
        )

        return results

    def remove_missing(self):

        remove_ids = [
            student_id
            for student_id, student
            in self.students.items()
            if student["missing"]
            > self.max_missing
        ]

        for student_id in remove_ids:

            del self.students[
                student_id
            ]


# ============================================================
# TEMPORAL ENGAGEMENT SMOOTHING
# ============================================================

class EngagementSmoother:

    def __init__(
        self,
        alpha=0.22,
        orientation_window=7,
        low_duration=1.5
    ):

        self.states = {}

        # Higher alpha = faster response.
        # Lower alpha = smoother output.
        self.alpha = alpha

        self.orientation_window = (
            orientation_window
        )

        # Student must remain below the low threshold
        # for this many seconds before being labelled Low.
        self.low_duration = low_duration

    def update(
        self,
        track_id,
        raw_score,
        raw_orientation
    ):

        now = time.monotonic()

        # ----------------------------------------------------
        # FIRST OBSERVATION
        # ----------------------------------------------------

        if track_id not in self.states:

            self.states[track_id] = {
                "score": float(raw_score),
                "orientations": deque(
                    maxlen=self.orientation_window
                ),
                "low_since": None,
                "last_seen": now
            }

        state = self.states[
            track_id
        ]

        state["last_seen"] = now

        # ----------------------------------------------------
        # SCORE SMOOTHING
        # ----------------------------------------------------

        old_score = state["score"]

        smoothed_score = (
            self.alpha
            * float(raw_score)
            +
            (1 - self.alpha)
            * old_score
        )

        state["score"] = smoothed_score

        display_score = round(
            smoothed_score
        )

        display_score = max(
            20,
            min(
                95,
                display_score
            )
        )

        # ----------------------------------------------------
        # ORIENTATION SMOOTHING
        # ----------------------------------------------------

        if (
            raw_orientation
            and raw_orientation
            != "Unknown"
        ):

            state[
                "orientations"
            ].append(
                raw_orientation
            )

        if state["orientations"]:

            stable_orientation = (
                Counter(
                    state[
                        "orientations"
                    ]
                )
                .most_common(1)[0][0]
            )

        else:

            stable_orientation = (
                "Unknown"
            )

        # ----------------------------------------------------
        # STATUS FROM SMOOTHED SCORE
        # ----------------------------------------------------

        if display_score >= 75:

            status = "Engaged"

            state["low_since"] = None

        elif display_score >= 55:

            status = "Neutral"

            state["low_since"] = None

        else:

            # The score is low, but don't immediately
            # label the student Low Engagement.
            if state["low_since"] is None:

                state["low_since"] = now

            low_time = (
                now
                - state["low_since"]
            )

            if (
                low_time
                >= self.low_duration
            ):

                status = (
                    "Low Engagement"
                )

            else:

                # Grace period for a short glance
                # down or sideways.
                status = "Neutral"

        return {
            "score": display_score,
            "status": status,
            "orientation": stable_orientation
        }

    def cleanup(self, tracker_students):

        existing_ids = set(
            tracker_students.keys()
        )

        remove_ids = [
            track_id
            for track_id
            in self.states
            if track_id not in existing_ids
        ]

        for track_id in remove_ids:

            del self.states[
                track_id
            ]


# ============================================================
# MATCH TRACKED FACE TO YUNET DETECTION
# ============================================================

def find_best_detection(
    student_face,
    faces,
    detections
):

    sx, sy, sw, sh = student_face

    student_centre = (
        sx + sw / 2,
        sy + sh / 2
    )

    best_detection = None
    best_distance = float(
        "inf"
    )

    for index, face in enumerate(
        faces
    ):

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

            if index < len(
                detections
            ):

                best_detection = (
                    detections[index]
                )

    return best_detection


# ============================================================
# CAMERA SOURCE SELECTION
# ============================================================

def select_camera_source():

    print()
    print(
        "======================================"
    )
    print(
        " CEMS CAMERA SOURCE"
    )
    print(
        "======================================"
    )
    print(
        "1 - USB Camera"
    )
    print(
        "2 - CCTV / IP Camera"
    )
    print(
        "3 - Video File"
    )
    print(
        "======================================"
    )
    print()

    while True:

        choice = input(
            "Select source (1, 2 or 3): "
        ).strip()

        # ----------------------------------------------------
        # USB
        # ----------------------------------------------------

        if choice == "1":

            return {
                "type": "usb",
                "source": 0,
                "name": "USB Camera",
                "min_face_size": 45
            }

        # ----------------------------------------------------
        # CCTV / RTSP
        # ----------------------------------------------------

        elif choice == "2":

            print()
            print(
                "Enter the CCTV/IP camera RTSP address."
            )
            print()
            print(
                "Do not save real CCTV passwords"
                " inside your Python source code."
            )
            print()

            rtsp_url = input(
                "RTSP URL: "
            ).strip()

            rtsp_url = (
                rtsp_url.strip(
                    "\"'"
                )
            )

            if not rtsp_url:

                print()
                print(
                    "No RTSP URL entered."
                )

                continue

            return {
                "type": "cctv",
                "source": rtsp_url,
                "name": "CCTV / IP Camera",
                "min_face_size": 30
            }

        # ----------------------------------------------------
        # VIDEO FILE
        # ----------------------------------------------------

        elif choice == "3":

            print()

            video_path = input(
                "Enter video file path: "
            ).strip()

            video_path = (
                video_path.strip(
                    "\"'"
                )
            )

            if not os.path.exists(
                video_path
            ):

                print()
                print(
                    "ERROR: Video file does not exist."
                )
                print()

                continue

            return {
                "type": "video",
                "source": video_path,
                "name": "Video File",
                "min_face_size": 30
            }

        else:

            print()
            print(
                "Please enter 1, 2 or 3."
            )
            print()


# ============================================================
# OPEN VIDEO SOURCE
# ============================================================

def open_video_source(
    source_info
):

    camera = cv2.VideoCapture(
        source_info["source"]
    )

    if not camera.isOpened():

        print()
        print(
            "ERROR: Could not open video source."
        )

        return None

    if (
        source_info["type"]
        == "usb"
    ):

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

    if (
        source_info["type"]
        == "cctv"
    ):

        # Reduce latency when supported
        # by the OpenCV backend.
        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    return camera


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "ERROR: YuNet model not found:"
        )

        print(
            MODEL_PATH
        )

        return

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    source_info = (
        select_camera_source()
    )

    camera = (
        open_video_source(
            source_info
        )
    )

    if camera is None:
        return

    # --------------------------------------------------------
    # YUNET
    # --------------------------------------------------------

    detector = (
        cv2.FaceDetectorYN.create(
            MODEL_PATH,
            "",
            (320, 320),
            0.75,
            0.3,
            5000
        )
    )

    # --------------------------------------------------------
    # TRACKER + SMOOTHER
    # --------------------------------------------------------

    tracker = StudentTracker(
        max_distance=140,
        max_missing=45
    )

    smoother = EngagementSmoother(
        alpha=0.22,
        orientation_window=7,
        low_duration=1.5
    )

    min_face_size = (
        source_info[
            "min_face_size"
        ]
    )

    # --------------------------------------------------------
    # VIDEO PLAYBACK DELAY
    # --------------------------------------------------------

    video_delay = 1

    if (
        source_info["type"]
        == "video"
    ):

        fps = camera.get(
            cv2.CAP_PROP_FPS
        )

        if fps > 1:

            video_delay = max(
                1,
                int(
                    1000 / fps
                )
            )

    print()
    print(
        "CEMS Engagement Monitoring started."
    )

    print(
        f"Source: {source_info['name']}"
    )

    print(
        "Temporal engagement smoothing enabled."
    )

    print(
        "Press Q to exit."
    )

    print()

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        success, frame = (
            camera.read()
        )

        if not success:

            if (
                source_info["type"]
                == "video"
            ):

                print(
                    "Video finished."
                )

            else:

                print(
                    "Could not read camera frame."
                )

            break

        height, width = (
            frame.shape[:2]
        )

        detector.setInputSize(
            (
                width,
                height
            )
        )

        _, detections = (
            detector.detect(
                frame
            )
        )

        faces = []

        valid_detections = []

        # ----------------------------------------------------
        # YUNET DETECTIONS
        # ----------------------------------------------------

        if detections is not None:

            for detection in detections:

                confidence = float(
                    detection[14]
                )

                if confidence < 0.75:
                    continue

                x, y, w, h = (
                    detection[:4]
                )

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                if (
                    w < min_face_size
                    or h < min_face_size
                ):

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

        # ----------------------------------------------------
        # TRACKING
        # ----------------------------------------------------

        tracked_students = (
            tracker.update(
                faces
            )
        )

        smoother.cleanup(
            tracker.students
        )

        engaged_count = 0
        neutral_count = 0
        low_count = 0

        # This structure remains compatible
        # with Deepesh's integration.
        live_results = []

        # ====================================================
        # EACH STUDENT
        # ====================================================

        for student in tracked_students:

            student_id = (
                student[
                    "student_id"
                ]
            )

            x, y, w, h = (
                student[
                    "face"
                ]
            )

            detection = (
                find_best_detection(
                    student["face"],
                    faces,
                    valid_detections
                )
            )

            # ------------------------------------------------
            # RAW ENGAGEMENT
            # ------------------------------------------------

            if detection is not None:

                raw_result = (
                    analyse_engagement(
                        detection
                    )
                )

                raw_score = (
                    raw_result[
                        "score"
                    ]
                )

                raw_orientation = (
                    raw_result[
                        "orientation"
                    ]
                )

            else:

                raw_score = 0
                raw_orientation = (
                    "Unknown"
                )

            # ------------------------------------------------
            # TEMPORAL FILTER
            # ------------------------------------------------

            if detection is not None:

                filtered = (
                    smoother.update(
                        student_id,
                        raw_score,
                        raw_orientation
                    )
                )

                score = (
                    filtered[
                        "score"
                    ]
                )

                status = (
                    filtered[
                        "status"
                    ]
                )

                orientation = (
                    filtered[
                        "orientation"
                    ]
                )

            else:

                score = 0
                status = "Unknown"
                orientation = "Unknown"

            # ------------------------------------------------
            # DATABASE / INTEGRATION OUTPUT
            # ------------------------------------------------

            student_result = {
                "track_id": student_id,
                "score": score,
                "status": status,
                "orientation": orientation
            }

            live_results.append(
                student_result
            )

            # ------------------------------------------------
            # COUNTS
            # ------------------------------------------------

            if status == "Engaged":

                engaged_count += 1

            elif status == "Neutral":

                neutral_count += 1

            elif (
                status
                == "Low Engagement"
            ):

                low_count += 1

            # ------------------------------------------------
            # COLOUR
            # ------------------------------------------------

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

            elif (
                status
                == "Low Engagement"
            ):

                box_colour = (
                    0,
                    0,
                    255
                )

            else:

                box_colour = (
                    180,
                    180,
                    180
                )

            # ------------------------------------------------
            # BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    x,
                    y
                ),
                (
                    x + w,
                    y + h
                ),
                box_colour,
                2
            )

            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

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

        # ====================================================
        # SUMMARY PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (
                10,
                10
            ),
            (
                420,
                90
            ),
            (
                0,
                0,
                0
            ),
            -1
        )

        cv2.putText(
            frame,
            (
                "Students Detected: "
                f"{len(tracked_students)}"
            ),
            (
                20,
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "Engaged: "
                f"{engaged_count}"
            ),
            (
                20,
                70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (
                0,
                255,
                0
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "Neutral: "
                f"{neutral_count}"
            ),
            (
                150,
                70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (
                0,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "Low: "
                f"{low_count}"
            ),
            (
                285,
                70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (
                0,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "CEMS - Engagement Monitoring",
            frame
        )

        key = (
            cv2.waitKey(
                video_delay
            )
            & 0xFF
        )

        if key == ord("q"):
            break

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()

    print()
    print(
        "CEMS session ended."
    )


if __name__ == "__main__":
    main()