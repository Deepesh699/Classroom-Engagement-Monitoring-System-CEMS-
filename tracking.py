import cv2
import joblib
import math
import os
import time 

from collections import Counter, deque

import pandas as pd

from engagement_features import (
    extract_frame_features,
    TemporalFeatureWindow
)
from face_recognition_service import FaceRecognitionService
from database import (
    get_active_session_id,
    get_student_by_id,
    get_track_assignments,
    assign_track_to_student,
    save_live_results
)


# ============================================================
# CONFIGURATION
# ============================================================

YUNET_MODEL_PATH = (
    "face_detection_yunet_2023mar.onnx"
)

ENGAGEMENT_MODEL_PATH = (
    "engagement_model_v2.joblib"
)

YUNET_CONFIDENCE = 0.75


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

    This is NOT face recognition.
    """

    def __init__(
        self,
        max_distance=140,
        max_missing=45
    ):
        self.students = {}

        self.max_distance = (
            max_distance
        )

        self.max_missing = (
            max_missing
        )

        self.next_student_id = 1

    # ========================================================
    # BOX CENTRE
    # ========================================================

    def centre(
        self,
        box
    ):
        x, y, w, h = box

        return (
            x + w / 2,
            y + h / 2
        )

    # ========================================================
    # DISTANCE
    # ========================================================

    def distance(
        self,
        p1,
        p2
    ):
        return math.hypot(
            p1[0] - p2[0],
            p1[1] - p2[1]
        )

    # ========================================================
    # IOU
    # ========================================================

    def iou(
        self,
        box1,
        box2
    ):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        left = max(
            x1,
            x2
        )

        top = max(
            y1,
            y2
        )

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
            *
            intersection_h
        )

        area1 = (
            max(
                0,
                w1
            )
            *
            max(
                0,
                h1
            )
        )

        area2 = (
            max(
                0,
                w2
            )
            *
            max(
                0,
                h2
            )
        )

        union = (
            area1
            +
            area2
            -
            intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection
            /
            union
        )

    # ========================================================
    # FACE SIZE SIMILARITY
    # ========================================================

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
            min(
                area1,
                area2
            )
            /
            max(
                area1,
                area2
            )
        )

    # ========================================================
    # PREDICT STUDENT POSITION
    # ========================================================

    def predict_box(
        self,
        student
    ):
        x, y, w, h = (
            student[
                "box"
            ]
        )

        vx, vy = (
            student[
                "velocity"
            ]
        )

        steps = max(
            1,
            min(
                student[
                    "missing"
                ],
                4
            )
        )

        predicted_x = (
            x
            +
            vx * steps
        )

        predicted_y = (
            y
            +
            vy * steps
        )

        return (
            predicted_x,
            predicted_y,
            w,
            h
        )

    # ========================================================
    # MOTION DIRECTION CONSISTENCY
    # ========================================================

    def motion_consistency(
        self,
        student,
        face
    ):
        vx, vy = (
            student[
                "velocity"
            ]
        )

        old_centre = (
            student[
                "centre"
            ]
        )

        new_centre = (
            self.centre(
                face
            )
        )

        dx = (
            new_centre[0]
            -
            old_centre[0]
        )

        dy = (
            new_centre[1]
            -
            old_centre[1]
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

        if (
            velocity_length < 2
            or
            movement_length < 2
        ):
            return 0.5

        cosine = (
            vx * dx
            +
            vy * dy
        ) / (
            velocity_length
            *
            movement_length
        )

        cosine = max(
            -1.0,
            min(
                1.0,
                cosine
            )
        )

        return (
            cosine + 1.0
        ) / 2.0

    # ========================================================
    # CREATE TRACK
    # ========================================================

    def create_student(
        self,
        face
    ):
        student_id = (
            self.next_student_id
        )

        self.next_student_id += 1

        self.students[
            student_id
        ] = {
            "box":
                face,

            "centre":
                self.centre(
                    face
                ),

            "velocity":
                (
                    0.0,
                    0.0
                ),

            "missing":
                0,

            "hits":
                1
        }

        return student_id

    # ========================================================
    # UPDATE TRACK
    # ========================================================

    def update_student(
        self,
        student_id,
        face
    ):
        student = (
            self.students[
                student_id
            ]
        )

        old_centre = (
            student[
                "centre"
            ]
        )

        new_centre = (
            self.centre(
                face
            )
        )

        frame_gap = max(
            1,
            student[
                "missing"
            ]
        )

        measured_vx = (
            new_centre[0]
            -
            old_centre[0]
        ) / frame_gap

        measured_vy = (
            new_centre[1]
            -
            old_centre[1]
        ) / frame_gap

        old_vx, old_vy = (
            student[
                "velocity"
            ]
        )

        student[
            "velocity"
        ] = (
            old_vx * 0.65
            +
            measured_vx * 0.35,

            old_vy * 0.65
            +
            measured_vy * 0.35
        )

        student[
            "box"
        ] = face

        student[
            "centre"
        ] = new_centre

        student[
            "missing"
        ] = 0

        student[
            "hits"
        ] += 1

    # ========================================================
    # MATCH SCORE
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

        face_centre = (
            self.centre(
                face
            )
        )

        centre_distance = (
            self.distance(
                predicted_centre,
                face_centre
            )
        )

        _, _, old_w, old_h = (
            student[
                "box"
            ]
        )

        _, _, new_w, new_h = (
            face
        )

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
                +
                new_diagonal
            ) / 2.0
        )

        normalized_distance = (
            centre_distance
            /
            face_scale
        )

        missing_bonus = (
            min(
                student[
                    "missing"
                ],
                10
            )
            *
            0.12
        )

        max_normalized_distance = (
            1.80
            +
            missing_bonus
        )

        absolute_distance_limit = (
            self.max_distance
            +
            min(
                student[
                    "missing"
                ],
                10
            )
            *
            12
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
            >
            max_normalized_distance

            and

            centre_distance
            >
            absolute_distance_limit

            and

            predicted_overlap
            <=
            0.01
        ):
            return None

        distance_score = max(
            0.0,
            1.0
            -
            (
                normalized_distance
                /
                max_normalized_distance
            )
        )

        size_score = (
            self.size_similarity(
                student[
                    "box"
                ],
                face
            )
        )

        motion_score = (
            self.motion_consistency(
                student,
                face
            )
        )

        # ----------------------------------------------------
        # MATCH WEIGHTS
        #
        # 45% predicted distance
        # 30% predicted IoU
        # 15% face-size similarity
        # 10% movement direction
        # ----------------------------------------------------

        match_score = (
            distance_score
            *
            0.45

            +

            predicted_overlap
            *
            0.30

            +

            size_score
            *
            0.15

            +

            motion_score
            *
            0.10
        )

        if match_score < 0.20:
            return None

        return match_score

    # ========================================================
    # MAIN TRACKER UPDATE
    # ========================================================

    def update(
        self,
        faces
    ):
        faces = [
            tuple(
                map(
                    int,
                    face
                )
            )
            for face in faces
        ]

        # Every student starts this frame as missing.
        # Successful matches reset missing to zero.
        for student in (
            self.students.values()
        ):
            student[
                "missing"
            ] += 1

        if not faces:

            self.remove_missing()

            return []

        candidates = []

        # ----------------------------------------------------
        # BUILD POSSIBLE MATCHES
        # ----------------------------------------------------

        for (
            student_id,
            student
        ) in self.students.items():

            for (
                face_index,
                face
            ) in enumerate(
                faces
            ):

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

        candidates.sort(
            key=lambda item:
                item[0],
            reverse=True
        )

        used_students = set()
        used_faces = set()

        visible_results = []

        # ----------------------------------------------------
        # ONE-TO-ONE MATCHING
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CREATE TRACKS FOR NEW FACES
        # ----------------------------------------------------

        for (
            face_index,
            face
        ) in enumerate(
            faces
        ):

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

    def remove_missing(
        self
    ):
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
# RANDOM FOREST ENGAGEMENT MANAGER
# ============================================================

class MLEngagementManager:
    """
    Maintains a separate temporal engagement history for
    every track_id.

    Example:

        Track 1 -> own 10-second TemporalFeatureWindow
        Track 2 -> own 10-second TemporalFeatureWindow
        Track 3 -> own 10-second TemporalFeatureWindow

    Behaviour from different students is never mixed.
    """

    def __init__(
        self,
        model_path
    ):

        if not os.path.exists(
            model_path
        ):
            raise FileNotFoundError(
                f"Engagement model not found: "
                f"{model_path}"
            )

        bundle = joblib.load(
            model_path
        )

        self.model = (
            bundle[
                "model"
            ]
        )

        # Live prediction does not need joblib workers.
        # This also prevents the parallel warning seen earlier.
        self.model.n_jobs = 1

        self.feature_columns = (
            bundle[
                "features"
            ]
        )

        self.window_seconds = float(
            bundle.get(
                "window_seconds",
                10.0
            )
        )

        self.windows = {}

        self.orientation_history = {}

        self.orientation_window = 7

    # ========================================================
    # GET / CREATE TEMPORAL WINDOW
    # ========================================================

    def get_window(
        self,
        track_id
    ):

        if (
            track_id
            not in self.windows
        ):
            self.windows[
                track_id
            ] = (
                TemporalFeatureWindow(
                    window_seconds=
                    self.window_seconds
                )
            )

        return self.windows[
            track_id
        ]

    # ========================================================
    # SMOOTH ORIENTATION
    # ========================================================

    def update_orientation(
        self,
        track_id,
        orientation
    ):

        if (
            track_id
            not in
            self.orientation_history
        ):
            self.orientation_history[
                track_id
            ] = deque(
                maxlen=
                self.orientation_window
            )

        history = (
            self.orientation_history[
                track_id
            ]
        )

        if (
            orientation
            and
            orientation
            != "Unknown"
        ):
            history.append(
                orientation
            )

        if not history:
            return "Unknown"

        return (
            Counter(
                history
            )
            .most_common(
                1
            )[0][0]
        )

    # ========================================================
    # DERIVED DISPLAY SCORE
    # ========================================================

    def calculate_score(
        self,
        probabilities
    ):
        probability_map = {}

        for (
            class_name,
            probability
        ) in zip(
            self.model.classes_,
            probabilities
        ):

            probability_map[
                str(
                    class_name
                )
            ] = float(
                probability
            )

        engaged = (
            probability_map.get(
                "Engaged",
                0.0
            )
        )

        neutral = (
            probability_map.get(
                "Neutral",
                0.0
            )
        )

        low = (
            probability_map.get(
                "Low Engagement",
                0.0
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This percentage is a derived DISPLAY score.
        #
        # The Random Forest itself predicts the engagement
        # class, not a continuous percentage.
        # ----------------------------------------------------

        score = (
            engaged
            *
            90.0

            +

            neutral
            *
            60.0

            +

            low
            *
            30.0
        )

        score = round(
            score
        )

        return max(
            20,
            min(
                95,
                score
            )
        )

    # ========================================================
    # ADD A MISSING-FACE FRAME
    # ========================================================

    def mark_missing(
        self,
        track_id
    ):

        window = (
            self.get_window(
                track_id
            )
        )

        window.add(
            None
        )

    # ========================================================
    # RECORD MISSING TRACKS
    # ========================================================

    def update_missing_tracks(
        self,
        tracker_students,
        visible_ids
    ):
        """
        A track may temporarily exist even though YuNet did
        not see its face in the current frame.

        Adding None lets face_visible_ratio capture that
        temporary face loss.
        """

        for track_id in (
            tracker_students.keys()
        ):

            if (
                track_id
                not in visible_ids
            ):
                self.mark_missing(
                    track_id
                )

    # ========================================================
    # PROCESS ONE VISIBLE TRACK
    # ========================================================

    def update(
        self,
        track_id,
        detection
    ):

        frame_features = (
            extract_frame_features(
                detection
            )
        )

        window = (
            self.get_window(
                track_id
            )
        )

        window.add(
            frame_features
        )

        orientation = (
            self.update_orientation(
                track_id,
                frame_features[
                    "orientation"
                ]
            )
        )

        window_duration = min(
            window.duration(),
            self.window_seconds
        )

        # ----------------------------------------------------
        # FIRST 10 SECONDS
        # ----------------------------------------------------

        if not window.ready():

            return {
                "score":
                    0,

                "status":
                    "Collecting",

                "orientation":
                    orientation,

                "confidence":
                    0.0,

                "ready":
                    False,

                "window_duration":
                    window_duration
            }

        # ----------------------------------------------------
        # TEMPORAL SUMMARY
        # ----------------------------------------------------

        summary = (
            window.summarize()
        )

        if summary is None:

            return {
                "score":
                    0,

                "status":
                    "Unknown",

                "orientation":
                    orientation,

                "confidence":
                    0.0,

                "ready":
                    False,

                "window_duration":
                    window_duration
            }

        # ----------------------------------------------------
        # EXACT FEATURE ORDER USED DURING TRAINING
        # ----------------------------------------------------

        model_input = {
            feature:
                summary[
                    feature
                ]

            for feature
            in self.feature_columns
        }

        input_df = pd.DataFrame(
            [
                model_input
            ],
            columns=
                self.feature_columns
        )

        # ----------------------------------------------------
        # RANDOM FOREST CLASSIFICATION
        # ----------------------------------------------------

        prediction = (
            self.model.predict(
                input_df
            )[0]
        )

        probabilities = (
            self.model.predict_proba(
                input_df
            )[0]
        )

        status = str(
            prediction
        )

        model_confidence = float(
            max(
                probabilities
            )
        )

        score = (
            self.calculate_score(
                probabilities
            )
        )

        return {
            "score":
                score,

            "status":
                status,

            "orientation":
                orientation,

            "confidence":
                model_confidence,

            "ready":
                True,

            "window_duration":
                window_duration
        }

    # ========================================================
    # REMOVE ML STATE FOR DELETED TRACKS
    # ========================================================

    def cleanup(
        self,
        tracker_students
    ):

        existing_ids = set(
            tracker_students.keys()
        )

        remove_windows = [
            track_id

            for track_id
            in self.windows

            if track_id
            not in existing_ids
        ]

        for track_id in remove_windows:

            del self.windows[
                track_id
            ]

        remove_orientations = [
            track_id

            for track_id
            in self.orientation_history

            if track_id
            not in existing_ids
        ]

        for track_id in (
            remove_orientations
        ):

            del self.orientation_history[
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

        # ====================================================
        # USB CAMERA
        # ====================================================

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

        # ====================================================
        # CCTV / RTSP
        # ====================================================

        elif choice == "2":

            print()

            print(
                "Enter the CCTV/IP camera RTSP address."
            )

            print(
                "Do not save real CCTV passwords "
                "inside the Python source code."
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

        # ====================================================
        # VIDEO FILE
        # ====================================================

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
# OPEN CAMERA / VIDEO
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

    # --------------------------------------------------------
    # USB CAMERA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CCTV
    # --------------------------------------------------------

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
# MAIN
# ============================================================

def main():

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(
        YUNET_MODEL_PATH
    ):

        print()

        print(
            "ERROR: YuNet model not found:"
        )

        print(
            YUNET_MODEL_PATH
        )

        return

    if not os.path.exists(
        ENGAGEMENT_MODEL_PATH
    ):

        print()

        print(
            "ERROR: Engagement model not found:"
        )

        print(
            ENGAGEMENT_MODEL_PATH
        )

        return

    # ========================================================
    # LOAD ML ENGAGEMENT MODEL
    # ========================================================

    try:

        engagement_manager = (
            MLEngagementManager(
                ENGAGEMENT_MODEL_PATH
            )
        )

    except Exception as error:

        print()

        print(
            "ERROR loading engagement model:"
        )

        print(
            error
        )

        return

    # ========================================================
    # CEMS STUDENT IDENTITY + SESSION INTEGRATION
    # ========================================================

    session_id = get_active_session_id()

    if session_id is None:

        print()
        print("ERROR: No active classroom session.")
        print("Start a session before running CEMS.")
        print()

        return

    print(
        f"Active classroom session: {session_id}"
    )

    try:

        face_recognition = (
            FaceRecognitionService(
                threshold=0.50
            )
        )

    except Exception as error:

        print()
        print(
            "ERROR loading SFace recognition:"
        )
        print(error)

        return

    if not face_recognition.registered_faces:

        print()
        print(
            "ERROR: No enrolled student faces found."
        )
        print(
            "Run face_registration.py first."
        )

        return

    print(
        "Registered face profiles:",
        len(
            face_recognition.registered_faces
        )
    )

    # Stable recognised identity for each temporary track.
    track_identity = {}

    # Require several consecutive matching recognition frames
    # before assigning a real student to a track.
    recognition_history = {}

    RECOGNITION_CONFIRM_FRAMES = 5

    # Save engagement periodically rather than every frame.
    SAVE_INTERVAL_SECONDS = 5

    last_database_save = time.time()

    # ========================================================
    # CAMERA SOURCE
    # ========================================================

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

    # ========================================================
    # YUNET
    # ========================================================

    detector = (
        cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (320, 320),
            YUNET_CONFIDENCE,
            0.3,
            5000
        )
    )

    # ========================================================
    # TRACKER
    # ========================================================

    tracker = (
        StudentTracker(
            max_distance=140,
            max_missing=45
        )
    )

    # If this program is restarted while the same classroom
    # session is still active, do not reuse old track IDs.
    existing_assignments = get_track_assignments(
        session_id
    )

    if existing_assignments:
        tracker.next_student_id = (
            max(
                assignment[1]
                for assignment in existing_assignments
            )
            + 1
        )

    min_face_size = (
        source_info[
            "min_face_size"
        ]
    )

    # ========================================================
    # VIDEO PLAYBACK SPEED
    # ========================================================

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

    # ========================================================
    # STARTUP INFORMATION
    # ========================================================

    print()

    print(
        "=========================================="
    )

    print(
        " CEMS MULTI-STUDENT ML ENGAGEMENT"
    )

    print(
        "=========================================="
    )

    print()

    print(
        f"Source: "
        f"{source_info['name']}"
    )

    print(
        "YuNet multi-face detection enabled."
    )

    print(
        "Multi-student tracking enabled."
    )

    print(
        "Random Forest engagement enabled."
    )

    print(
        (
            "Independent temporal window per track: "
            f"{engagement_manager.window_seconds:.0f} seconds"
        )
    )

    print()

    print(
        "Track IDs are temporary session IDs."
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

        # ----------------------------------------------------
        # YUNET INPUT SIZE
        # ----------------------------------------------------

        detector.setInputSize(
            (
                width,
                height
            )
        )

        # ----------------------------------------------------
        # DETECT FACES
        # ----------------------------------------------------

        _, detections = (
            detector.detect(
                frame
            )
        )

        faces = []

        valid_detections = []

        if detections is not None:

            for detection in (
                detections
            ):

                confidence = float(
                    detection[
                        14
                    ]
                )

                if (
                    confidence
                    <
                    YUNET_CONFIDENCE
                ):
                    continue

                x, y, w, h = (
                    detection[
                        :4
                    ]
                )

                x = int(
                    x
                )

                y = int(
                    y
                )

                w = int(
                    w
                )

                h = int(
                    h
                )

                if (
                    w
                    <
                    min_face_size

                    or

                    h
                    <
                    min_face_size
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

                # IMPORTANT:
                #
                # This list stays in exactly the same order
                # as faces.
                #
                # Therefore StudentTracker.face_index can be
                # used to retrieve the correct YuNet landmarks.
                valid_detections.append(
                    detection
                )

        # ====================================================
        # MULTI-STUDENT TRACKING
        # ========================================================

        tracked_students = (
            tracker.update(
                faces
            )
        )

        visible_ids = {
            student[
                "student_id"
            ]
            for student
            in tracked_students
        }

        # ----------------------------------------------------
        # RECORD TEMPORARY FACE LOSS
        # ----------------------------------------------------

        engagement_manager.update_missing_tracks(
            tracker.students,
            visible_ids
        )

        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        engaged_count = 0

        neutral_count = 0

        low_count = 0

        collecting_count = 0

        # ----------------------------------------------------
        # IMPORTANT TEAM OUTPUT
        #
        # Keep these four fields unchanged for Deepesh.
        # ----------------------------------------------------

        live_results = []

        # ====================================================
        # PROCESS EVERY VISIBLE STUDENT
        # ========================================================

        for student in (
            tracked_students
        ):

            track_id = (
                student[
                    "student_id"
                ]
            )

            x, y, w, h = (
                student[
                    "face"
                ]
            )

            face_index = (
                student[
                    "face_index"
                ]
            )

            detection = None

            if (
                0
                <=
                face_index
                <
                len(
                    valid_detections
                )
            ):

                detection = (
                    valid_detections[
                        face_index
                    ]
                )

            # =================================================
            # ML ENGAGEMENT FOR THIS TRACK ONLY
            # =================================================

            if detection is not None:

                ml_result = (
                    engagement_manager.update(
                        track_id,
                        detection
                    )
                )

                score = (
                    ml_result[
                        "score"
                    ]
                )

                status = (
                    ml_result[
                        "status"
                    ]
                )

                orientation = (
                    ml_result[
                        "orientation"
                    ]
                )

                model_confidence = (
                    ml_result[
                        "confidence"
                    ]
                )

                ready = (
                    ml_result[
                        "ready"
                    ]
                )

                window_duration = (
                    ml_result[
                        "window_duration"
                    ]
                )

            else:

                # This should be unusual because face_index
                # normally maps directly to valid_detections.
                engagement_manager.mark_missing(
                    track_id
                )

                score = 0

                status = (
                    "Unknown"
                )

                orientation = (
                    "Unknown"
                )

                model_confidence = 0.0

                ready = False

                window_duration = 0.0

            # =================================================
            # REGISTERED STUDENT IDENTITY
            # =================================================

            registered_student_id = (
                track_identity.get(
                    track_id
                )
            )

            # -------------------------------------------------
            # FACE RECOGNITION
            # -------------------------------------------------

            if (
                detection is not None
                and registered_student_id is None
            ):

                try:

                    feature = (
                        face_recognition.extract_feature(
                            frame,
                            detection
                        )
                    )

                    recognition_result = (
                        face_recognition.recognise(
                            feature
                        )
                    )

                    if recognition_result[
                        "recognised"
                    ]:

                        recognised_id = (
                            recognition_result[
                                "student_id"
                            ]
                        )

                        # One registered student can belong to only
                        # one active track at a time. This prevents
                        # two faces being labelled as the same person.
                        already_assigned = (
                            recognised_id
                            in track_identity.values()
                        )

                        if already_assigned:

                            if track_id in recognition_history:
                                recognition_history[
                                    track_id
                                ].clear()

                        else:

                            if track_id not in recognition_history:

                                recognition_history[
                                    track_id
                                ] = deque(
                                    maxlen=(
                                        RECOGNITION_CONFIRM_FRAMES
                                    )
                                )

                            recognition_history[
                                track_id
                            ].append(
                                recognised_id
                            )

                            history = list(
                                recognition_history[
                                    track_id
                                ]
                            )

                            if (
                                len(history)
                                ==
                                RECOGNITION_CONFIRM_FRAMES
                                and
                                len(set(history))
                                == 1
                            ):

                                confirmed_student_id = (
                                    history[0]
                                )

                                student_record = (
                                    get_student_by_id(
                                        confirmed_student_id
                                    )
                                )

                                if student_record is not None:

                                    registered_student_id = (
                                        confirmed_student_id
                                    )

                                    track_identity[
                                        track_id
                                    ] = (
                                        confirmed_student_id
                                    )

                                    assign_track_to_student(
                                        session_id,
                                        track_id,
                                        confirmed_student_id
                                    )

                                    print(
                                        f"Track {track_id} "
                                        f"recognised as "
                                        f"{student_record[2]} "
                                        f"({student_record[1]})"
                                    )

                    else:

                        if track_id in recognition_history:

                            recognition_history[
                                track_id
                            ].clear()

                except cv2.error as error:

                    print(
                        "Face recognition error:",
                        error
                    )

            # =================================================
            # STABLE TEAM / DATABASE OUTPUT CONTRACT
            # =================================================

            student_result = {
                "track_id":
                    track_id,

                "score":
                    score,

                "status":
                    status,

                "orientation":
                    orientation
            }

            if registered_student_id is not None:

                student_result[
                    "registered_student_id"
                ] = (
                    registered_student_id
                )

            live_results.append(
                student_result
            )

            # =================================================
            # SUMMARY COUNTS
            # =================================================

            if status == "Engaged":

                engaged_count += 1

            elif status == "Neutral":

                neutral_count += 1

            elif (
                status
                ==
                "Low Engagement"
            ):

                low_count += 1

            elif (
                status
                ==
                "Collecting"
            ):

                collecting_count += 1

            # =================================================
            # BOX COLOUR
            # =================================================

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
                ==
                "Low Engagement"
            ):

                box_colour = (
                    0,
                    0,
                    255
                )

            elif status == "Collecting":

                box_colour = (
                    255,
                    180,
                    0
                )

            else:

                box_colour = (
                    180,
                    180,
                    180
                )

            # =================================================
            # FACE BOX
            # =================================================

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

            # =================================================
            # STUDENT LABEL
            # =================================================

            student_name = None
            student_number = None

            if registered_student_id is not None:

                student_record = get_student_by_id(
                    registered_student_id
                )

                if student_record is not None:

                    student_number = student_record[1]
                    student_name = student_record[2]

            if student_name is not None:

                identity_text = (
                    f"{student_name} | "
                    f"{student_number}"
                )

            else:

                identity_text = (
                    f"Unknown Student | "
                    f"Track {track_id}"
                )

            if ready:

                label = (
                    f"{identity_text} | "
                    f"{orientation} | "
                    f"{status} | "
                    f"{score}% | "
                    f"Conf {model_confidence:.0%}"
                )

            else:

                label = (
                    f"{identity_text} | "
                    f"{orientation} | "
                    f"Collecting "
                    f"{window_duration:.1f}/"
                    f"{engagement_manager.window_seconds:.0f}s"
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
                0.50,
                box_colour,
                2,
                cv2.LINE_AA
            )

        # ====================================================
        # CLEAN UP ENGAGEMENT HISTORIES FOR DELETED TRACKS
        # ====================================================

        engagement_manager.cleanup(
            tracker.students
        )

        # ====================================================
        # CLEAN UP IDENTITY STATE FOR DELETED TRACKS
        # ====================================================

        active_track_ids = set(
            tracker.students.keys()
        )

        for old_track_id in list(
            track_identity.keys()
        ):

            if old_track_id not in active_track_ids:
                del track_identity[
                    old_track_id
                ]

        for old_track_id in list(
            recognition_history.keys()
        ):

            if old_track_id not in active_track_ids:
                del recognition_history[
                    old_track_id
                ]

        # ====================================================
        # SAVE REGISTERED STUDENT ENGAGEMENT TO SQLITE
        # ====================================================

        current_time = time.time()

        if (
            live_results
            and
            current_time - last_database_save >= SAVE_INTERVAL_SECONDS
        ):

            # Keep live_results unchanged for team integration,
            # but only persist meaningful, recognised ML results.
            database_results = [
                result
                for result in live_results
                if (
                    result.get(
                        "registered_student_id"
                    )
                    is not None
                    and
                    result.get(
                        "status"
                    )
                    not in (
                        "Collecting",
                        "Unknown"
                    )
                )
            ]

            saved_count = 0

            if database_results:

                saved_count = save_live_results(
                    session_id,
                    database_results
                )

            if saved_count > 0:
                print(
                    f"Saved {saved_count} "
                    f"registered engagement record(s) "
                    f"for session {session_id}"
                )

            last_database_save = current_time

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
                520,
                120
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
                38
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
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
            0.55,
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
                145,
                70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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
                270,
                70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "Collecting: "
                f"{collecting_count}"
            ),
            (
                360,
                70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                180,
                0
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "ML Window: "
                f"{engagement_manager.window_seconds:.0f}s "
                "| Independent per track_id"
            ),
            (
                20,
                102
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            (
                "CEMS - Multi-Student "
                "ML Engagement Monitoring"
            ),
            frame
        )

        key = (
            cv2.waitKey(
                video_delay
            )
            &
            0xFF
        )

        if key == ord(
            "q"
        ):
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

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()