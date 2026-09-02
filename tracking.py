import cv2
import math
import os
import time
from collections import Counter, deque

from engagement_service import analyse_engagement


MODEL_PATH = "face_detection_yunet_2023mar.onnx"


# ============================================================
# MULTI-STUDENT TRACKER
# ============================================================

class StudentTracker:
    """
    Geometry-based multi-student tracker.

    Matching uses:
    - predicted movement
    - normalized centre distance
    - predicted bounding-box IoU
    - face-size similarity
    - movement-direction consistency

    Track IDs are temporary session IDs.
    This is not facial recognition.
    """

    def __init__(self, max_distance=140, max_missing=45):
        self.students = {}
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_student_id = 1

    def centre(self, box):
        x, y, w, h = box

        return (
            x + w / 2,
            y + h / 2
        )

    def distance(self, p1, p2):
        return math.hypot(
            p1[0] - p2[0],
            p1[1] - p2[1]
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

        intersection_w = max(
            0,
            right - left
        )

        intersection_h = max(
            0,
            bottom - top
        )

        intersection = (
            intersection_w
            * intersection_h
        )

        area1 = (
            max(0, w1)
            * max(0, h1)
        )

        area2 = (
            max(0, w2)
            * max(0, h2)
        )

        union = (
            area1
            + area2
            - intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection
            / union
        )

    def size_similarity(
        self,
        box1,
        box2
    ):
        _, _, w1, h1 = box1
        _, _, w2, h2 = box2

        area1 = max(
            1,
            w1 * h1
        )

        area2 = max(
            1,
            w2 * h2
        )

        return (
            min(area1, area2)
            / max(area1, area2)
        )

    # ========================================================
    # PREDICT WHERE A STUDENT SHOULD MOVE
    # ========================================================

    def predict_box(self, student):
        x, y, w, h = student["box"]

        vx, vy = student[
            "velocity"
        ]

        # If a student disappears briefly,
        # predict slightly farther forward.
        steps = max(
            1,
            min(
                student["missing"],
                4
            )
        )

        predicted_x = (
            x + vx * steps
        )

        predicted_y = (
            y + vy * steps
        )

        return (
            predicted_x,
            predicted_y,
            w,
            h
        )

    # ========================================================
    # CHECK IF MOVEMENT DIRECTION MAKES SENSE
    # ========================================================

    def motion_consistency(
        self,
        student,
        face
    ):
        vx, vy = student[
            "velocity"
        ]

        old_centre = student[
            "centre"
        ]

        new_centre = self.centre(
            face
        )

        dx = (
            new_centre[0]
            - old_centre[0]
        )

        dy = (
            new_centre[1]
            - old_centre[1]
        )

        velocity_length = (
            math.hypot(
                vx,
                vy
            )
        )

        movement_length = (
            math.hypot(
                dx,
                dy
            )
        )

        # If the student is barely moving,
        # use a neutral motion score.
        if (
            velocity_length < 2
            or movement_length < 2
        ):
            return 0.5

        cosine = (
            vx * dx
            + vy * dy
        ) / (
            velocity_length
            * movement_length
        )

        cosine = max(
            -1.0,
            min(
                1.0,
                cosine
            )
        )

        # Convert -1..1 to 0..1.
        return (
            cosine + 1.0
        ) / 2.0

    # ========================================================
    # CREATE NEW STUDENT
    # ========================================================

    def create_student(self, face):
        student_id = (
            self.next_student_id
        )

        self.next_student_id += 1

        self.students[
            student_id
        ] = {
            "box": face,
            "centre": self.centre(
                face
            ),
            "velocity": (
                0.0,
                0.0
            ),
            "missing": 0,
            "hits": 1
        }

        return student_id

    # ========================================================
    # UPDATE EXISTING STUDENT
    # ========================================================

    def update_student(
        self,
        student_id,
        face
    ):
        student = self.students[
            student_id
        ]

        old_centre = student[
            "centre"
        ]

        new_centre = self.centre(
            face
        )

        # missing is incremented before matching,
        # so this also gives us the frame gap.
        frame_gap = max(
            1,
            student["missing"]
        )

        measured_vx = (
            new_centre[0]
            - old_centre[0]
        ) / frame_gap

        measured_vy = (
            new_centre[1]
            - old_centre[1]
        ) / frame_gap

        old_vx, old_vy = student[
            "velocity"
        ]

        # Smooth velocity.
        student["velocity"] = (
            old_vx * 0.65
            + measured_vx * 0.35,

            old_vy * 0.65
            + measured_vy * 0.35
        )

        student["box"] = face

        student["centre"] = (
            new_centre
        )

        student["missing"] = 0

        student["hits"] += 1

    # ========================================================
    # CALCULATE MATCH SCORE
    # ========================================================

    def build_match_score(
        self,
        student,
        face
    ):
        predicted_box = (
            self.predict_box(
                student
            )
        )

        predicted_centre = (
            self.centre(
                predicted_box
            )
        )

        face_centre = self.centre(
            face
        )

        centre_distance = (
            self.distance(
                predicted_centre,
                face_centre
            )
        )

        _, _, old_w, old_h = (
            student["box"]
        )

        _, _, new_w, new_h = face

        old_diagonal = (
            math.hypot(
                old_w,
                old_h
            )
        )

        new_diagonal = (
            math.hypot(
                new_w,
                new_h
            )
        )

        face_scale = max(
            1.0,
            (
                old_diagonal
                + new_diagonal
            ) / 2.0
        )

        # Normalize movement relative
        # to face size.
        normalized_distance = (
            centre_distance
            / face_scale
        )

        # If a face was missing briefly,
        # allow a slightly larger search area.
        missing_bonus = (
            min(
                student["missing"],
                10
            )
            * 0.12
        )

        max_normalized_distance = (
            1.80
            + missing_bonus
        )

        absolute_distance_limit = (
            self.max_distance
            +
            min(
                student["missing"],
                10
            )
            * 12
        )

        predicted_overlap = (
            self.iou(
                predicted_box,
                face
            )
        )

        # Reject obviously impossible matches.
        if (
            normalized_distance
            > max_normalized_distance
            and centre_distance
            > absolute_distance_limit
            and predicted_overlap
            <= 0.01
        ):
            return None

        distance_score = max(
            0.0,
            1.0
            - (
                normalized_distance
                / max_normalized_distance
            )
        )

        size_score = (
            self.size_similarity(
                student["box"],
                face
            )
        )

        motion_score = (
            self.motion_consistency(
                student,
                face
            )
        )

        # ----------------------------------------
        # TRACKING MATCH WEIGHTS
        #
        # 45% predicted distance
        # 30% predicted IoU
        # 15% face-size similarity
        # 10% movement direction
        # ----------------------------------------

        match_score = (
            distance_score
            * 0.45
            +
            predicted_overlap
            * 0.30
            +
            size_score
            * 0.15
            +
            motion_score
            * 0.10
        )

        # Prevent weak matches from stealing IDs.
        if match_score < 0.20:
            return None

        return match_score

    # ========================================================
    # UPDATE TRACKER
    # ========================================================

    def update(self, faces):
        faces = [
            tuple(
                map(
                    int,
                    face
                )
            )
            for face in faces
        ]

        # Every track is considered missing
        # until matched in this frame.
        for student in (
            self.students.values()
        ):
            student["missing"] += 1

        if not faces:
            self.remove_missing()
            return []

        candidates = []

        # ====================================================
        # BUILD ALL POSSIBLE MATCHES
        # ====================================================

        for (
            student_id,
            student
        ) in self.students.items():

            for (
                face_index,
                face
            ) in enumerate(faces):

                match_score = (
                    self.build_match_score(
                        student,
                        face
                    )
                )

                if (
                    match_score
                    is not None
                ):
                    candidates.append(
                        (
                            match_score,
                            student_id,
                            face_index
                        )
                    )

        # Strongest matches first.
        candidates.sort(
            key=lambda item:
            item[0],
            reverse=True
        )

        used_students = set()
        used_faces = set()

        visible_results = []

        # ====================================================
        # ONE-TO-ONE MATCHING
        # ====================================================

        for (
            match_score,
            student_id,
            face_index
        ) in candidates:

            if (
                student_id
                in used_students
            ):
                continue

            if (
                face_index
                in used_faces
            ):
                continue

            self.update_student(
                student_id,
                faces[
                    face_index
                ]
            )

            used_students.add(
                student_id
            )

            used_faces.add(
                face_index
            )

            visible_results.append(
                {
                    "student_id":
                        student_id,

                    "face":
                        faces[
                            face_index
                        ],

                    "face_index":
                        face_index,

                    "match_score":
                        round(
                            match_score,
                            3
                        )
                }
            )

        # ====================================================
        # NEW FACES
        # ====================================================

        for (
            face_index,
            face
        ) in enumerate(faces):

            if (
                face_index
                in used_faces
            ):
                continue

            student_id = (
                self.create_student(
                    face
                )
            )

            visible_results.append(
                {
                    "student_id":
                        student_id,

                    "face":
                        face,

                    "face_index":
                        face_index,

                    "match_score":
                        1.0
                }
            )

        self.remove_missing()

        visible_results.sort(
            key=lambda item:
            item[
                "student_id"
            ]
        )

        return visible_results

    # ========================================================
    # REMOVE OLD TRACKS
    # ========================================================

    def remove_missing(self):
        remove_ids = [
            student_id

            for (
                student_id,
                student
            ) in self.students.items()

            if student[
                "missing"
            ] > self.max_missing
        ]

        for student_id in remove_ids:
            del self.students[
                student_id
            ]


# ============================================================
# ENGAGEMENT SMOOTHING
# ============================================================

class EngagementSmoother:

    def __init__(
        self,
        alpha=0.22,
        orientation_window=7,
        low_duration=1.5
    ):
        self.states = {}

        self.alpha = alpha

        self.orientation_window = (
            orientation_window
        )

        self.low_duration = (
            low_duration
        )

    def update(
        self,
        track_id,
        raw_score,
        raw_orientation
    ):
        now = time.monotonic()

        if (
            track_id
            not in self.states
        ):
            self.states[
                track_id
            ] = {
                "score":
                    float(
                        raw_score
                    ),

                "orientations":
                    deque(
                        maxlen=
                        self.orientation_window
                    ),

                "low_since":
                    None,

                "last_seen":
                    now
            }

        state = self.states[
            track_id
        ]

        state["last_seen"] = now

        old_score = state[
            "score"
        ]

        smoothed_score = (
            self.alpha
            * float(
                raw_score
            )
            +
            (
                1
                - self.alpha
            )
            * old_score
        )

        state["score"] = (
            smoothed_score
        )

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

        if state[
            "orientations"
        ]:
            stable_orientation = (
                Counter(
                    state[
                        "orientations"
                    ]
                )
                .most_common(
                    1
                )[0][0]
            )

        else:
            stable_orientation = (
                "Unknown"
            )

        # ----------------------------------------------------
        # ENGAGEMENT STATUS
        # ----------------------------------------------------

        if display_score >= 75:

            status = "Engaged"

            state[
                "low_since"
            ] = None

        elif display_score >= 55:

            status = "Neutral"

            state[
                "low_since"
            ] = None

        else:

            if (
                state[
                    "low_since"
                ]
                is None
            ):
                state[
                    "low_since"
                ] = now

            low_time = (
                now
                - state[
                    "low_since"
                ]
            )

            if (
                low_time
                >= self.low_duration
            ):
                status = (
                    "Low Engagement"
                )

            else:
                status = (
                    "Neutral"
                )

        return {
            "score":
                display_score,

            "status":
                status,

            "orientation":
                stable_orientation
        }

    def cleanup(
        self,
        tracker_students
    ):
        existing_ids = set(
            tracker_students.keys()
        )

        remove_ids = [
            track_id

            for track_id
            in self.states

            if track_id
            not in existing_ids
        ]

        for track_id in remove_ids:

            del self.states[
                track_id
            ]


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
        # USB CAMERA
        # ----------------------------------------------------

        if choice == "1":

            return {
                "type":
                    "usb",

                "source":
                    0,

                "name":
                    "USB Camera",

                "min_face_size":
                    45
            }

        # ----------------------------------------------------
        # CCTV / RTSP
        # ----------------------------------------------------

        elif choice == "2":

            print()

            print(
                "Enter the CCTV/IP camera RTSP address."
            )

            print(
                "Do not save real CCTV passwords "
                "inside your Python source code."
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
                "type":
                    "cctv",

                "source":
                    rtsp_url,

                "name":
                    "CCTV / IP Camera",

                "min_face_size":
                    30
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
                "type":
                    "video",

                "source":
                    video_path,

                "name":
                    "Video File",

                "min_face_size":
                    30
            }

        else:

            print()

            print(
                "Please enter 1, 2 or 3."
            )

            print()


# ============================================================
# OPEN CAMERA / VIDEO SOURCE
# ============================================================

def open_video_source(
    source_info
):

    camera = cv2.VideoCapture(
        source_info[
            "source"
        ]
    )

    if not camera.isOpened():

        print()

        print(
            "ERROR: Could not open video source."
        )

        return None

    if (
        source_info[
            "type"
        ] == "usb"
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
        source_info[
            "type"
        ] == "cctv"
    ):

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    return camera


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK YUNET MODEL
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
    # SELECT CAMERA
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
    # YUNET DETECTOR
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
    # TRACKER
    # --------------------------------------------------------

    tracker = StudentTracker(
        max_distance=140,
        max_missing=45
    )

    # --------------------------------------------------------
    # ENGAGEMENT SMOOTHER
    # --------------------------------------------------------

    smoother = (
        EngagementSmoother(
            alpha=0.22,
            orientation_window=7,
            low_duration=1.5
        )
    )

    min_face_size = (
        source_info[
            "min_face_size"
        ]
    )

    # --------------------------------------------------------
    # VIDEO PLAYBACK SPEED
    # --------------------------------------------------------

    video_delay = 1

    if (
        source_info[
            "type"
        ] == "video"
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
        "Improved multi-student tracking enabled."
    )

    print(
        "Temporal engagement smoothing enabled."
    )

    print(
        "Press Q to exit."
    )

    print()

    # ========================================================
    # MAIN FRAME LOOP
    # ========================================================

    while True:

        success, frame = (
            camera.read()
        )

        if not success:

            if (
                source_info[
                    "type"
                ] == "video"
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
        # YUNET FACE DETECTION
        # ----------------------------------------------------

        if (
            detections
            is not None
        ):

            for detection in detections:

                confidence = float(
                    detection[
                        14
                    ]
                )

                if (
                    confidence
                    < 0.75
                ):
                    continue

                x, y, w, h = (
                    detection[
                        :4
                    ]
                )

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                if (
                    w
                    < min_face_size
                    or h
                    < min_face_size
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
        # MULTI-STUDENT TRACKING
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

        # Keep this unchanged for Deepesh/database.
        live_results = []

        # ====================================================
        # PROCESS EACH STUDENT
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

            # Because tracker now preserves the
            # original face index, we do not need
            # to guess which YuNet detection belongs
            # to which student.
            face_index = (
                student[
                    "face_index"
                ]
            )

            detection = None

            if (
                0
                <= face_index
                < len(
                    valid_detections
                )
            ):

                detection = (
                    valid_detections[
                        face_index
                    ]
                )

            # ------------------------------------------------
            # ENGAGEMENT ANALYSIS
            # ------------------------------------------------

            if (
                detection
                is not None
            ):

                raw_result = (
                    analyse_engagement(
                        detection
                    )
                )

                filtered = (
                    smoother.update(
                        student_id,
                        raw_result[
                            "score"
                        ],
                        raw_result[
                            "orientation"
                        ]
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

                status = (
                    "Unknown"
                )

                orientation = (
                    "Unknown"
                )

            # ------------------------------------------------
            # DATABASE / TEAM INTEGRATION OUTPUT
            # ------------------------------------------------

            student_result = {
                "track_id":
                    student_id,

                "score":
                    score,

                "status":
                    status,

                "orientation":
                    orientation
            }

            live_results.append(
                student_result
            )

            # ------------------------------------------------
            # SUMMARY COUNTS
            # ------------------------------------------------

            if (
                status
                == "Engaged"
            ):

                engaged_count += 1

            elif (
                status
                == "Neutral"
            ):

                neutral_count += 1

            elif (
                status
                == "Low Engagement"
            ):

                low_count += 1

            # ------------------------------------------------
            # BOX COLOUR
            # ------------------------------------------------

            if (
                status
                == "Engaged"
            ):

                box_colour = (
                    0,
                    255,
                    0
                )

            elif (
                status
                == "Neutral"
            ):

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
            # FACE BOX
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
            # STUDENT LABEL
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