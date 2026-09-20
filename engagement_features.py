import math
import time
from collections import deque

import numpy as np


# ============================================================
# SINGLE-FRAME FEATURE EXTRACTION
# ============================================================

def extract_frame_features(detection):
    """
    Extract observable behavioural features from one
    YuNet face detection.

    YuNet detection format:

        0-3   = face bounding box
        4-5   = right eye
        6-7   = left eye
        8-9   = nose
        10-11 = right mouth corner
        12-13 = left mouth corner
        14    = face detection confidence
    """

    # --------------------------------------------------------
    # FACE BOUNDING BOX
    # --------------------------------------------------------

    x = float(detection[0])
    y = float(detection[1])
    w = float(detection[2])
    h = float(detection[3])

    # --------------------------------------------------------
    # LANDMARKS
    # --------------------------------------------------------

    right_eye = (
        float(detection[4]),
        float(detection[5])
    )

    left_eye = (
        float(detection[6]),
        float(detection[7])
    )

    nose = (
        float(detection[8]),
        float(detection[9])
    )

    right_mouth = (
        float(detection[10]),
        float(detection[11])
    )

    left_mouth = (
        float(detection[12]),
        float(detection[13])
    )

    confidence = float(
        detection[14]
    )

    # ========================================================
    # LANDMARK MIDPOINTS
    # ========================================================

    eye_mid_x = (
        right_eye[0] + left_eye[0]
    ) / 2.0

    eye_mid_y = (
        right_eye[1] + left_eye[1]
    ) / 2.0

    mouth_mid_x = (
        right_mouth[0] + left_mouth[0]
    ) / 2.0

    mouth_mid_y = (
        right_mouth[1] + left_mouth[1]
    ) / 2.0

    # ========================================================
    # NORMALISING DISTANCES
    # ========================================================

    eye_distance = math.hypot(
        left_eye[0] - right_eye[0],
        left_eye[1] - right_eye[1]
    )

    eye_distance = max(
        eye_distance,
        1.0
    )

    face_vertical = math.hypot(
        mouth_mid_x - eye_mid_x,
        mouth_mid_y - eye_mid_y
    )

    face_vertical = max(
        face_vertical,
        1.0
    )

    face_width = max(
        w,
        1.0
    )

    # ========================================================
    # YAW
    #
    # Horizontal head direction.
    # ========================================================

    yaw = (
        nose[0] - eye_mid_x
    ) / eye_distance

    # ========================================================
    # PITCH
    #
    # Vertical head direction.
    # ========================================================

    pitch = (
        nose[1] - eye_mid_y
    ) / face_vertical

    # ========================================================
    # ROLL
    #
    # Sideways head tilt.
    # ========================================================

    roll_radians = math.atan2(
        left_eye[1] - right_eye[1],
        left_eye[0] - right_eye[0]
    )

    roll = math.degrees(
        roll_radians
    )

    # ========================================================
    # FACE GEOMETRY
    # ========================================================

    eye_distance_ratio = (
        eye_distance / face_width
    )

    # ========================================================
    # LOOKING DOWN
    # ========================================================

    looking_down = (
        pitch > 0.72
    )

    # Continuous downward severity.
    #
    # Around 0.0 = not meaningfully downward
    # Around 1.0 = strongly downward
    #
    downward_severity = (
        pitch - 0.60
    ) / 0.35

    downward_severity = max(
        0.0,
        min(
            1.0,
            downward_severity
        )
    )

    # ========================================================
    # FORWARD / AWAY
    # ========================================================

    roughly_forward = (
        abs(yaw) <= 0.32
        and
        0.32 <= pitch <= 0.72
    )

    away = (
        abs(yaw) > 0.32
        or
        pitch > 0.72
        or
        pitch < 0.32
    )

    # ========================================================
    # CURRENT HEAD ORIENTATION
    # ========================================================

    if yaw < -0.32:
        orientation = "Right"

    elif yaw > 0.32:
        orientation = "Left"

    elif pitch > 0.72:
        orientation = "Down"

    elif pitch < 0.32:
        orientation = "Up"

    else:
        orientation = "Forward"

    # ========================================================
    # RETURN CURRENT FRAME FEATURES
    # ========================================================

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "confidence": confidence,
        "eye_distance_ratio": eye_distance_ratio,
        "looking_down": looking_down,
        "downward_severity": downward_severity,
        "roughly_forward": roughly_forward,
        "away": away,
        "orientation": orientation
    }


# ============================================================
# TEMPORAL FEATURE WINDOW
# ============================================================

class TemporalFeatureWindow:
    """
    Stores the most recent 10 seconds of behaviour.

    The Random Forest does not classify an individual frame.

    Instead, it receives a summary of behaviour observed across
    the temporal window.
    """

    def __init__(
        self,
        window_seconds=10.0
    ):
        self.window_seconds = float(
            window_seconds
        )

        self.samples = deque()

    # --------------------------------------------------------
    # CLEAR ALL HISTORY
    # --------------------------------------------------------

    def clear(self):
        self.samples.clear()

    # --------------------------------------------------------
    # ADD ONE FRAME
    # --------------------------------------------------------

    def add(
        self,
        features
    ):
        now = time.monotonic()

        self.samples.append(
            {
                "time": now,
                "features": features
            }
        )

        self._remove_old(
            now
        )

    # --------------------------------------------------------
    # REMOVE FRAMES OLDER THAN WINDOW
    # --------------------------------------------------------

    def _remove_old(
        self,
        now
    ):
        cutoff = (
            now
            - self.window_seconds
        )

        while (
            self.samples
            and
            self.samples[0]["time"] < cutoff
        ):
            self.samples.popleft()

    # --------------------------------------------------------
    # CURRENT WINDOW DURATION
    # --------------------------------------------------------

    def duration(self):
        if len(self.samples) < 2:
            return 0.0

        return (
            self.samples[-1]["time"]
            -
            self.samples[0]["time"]
        )

    # --------------------------------------------------------
    # IS THE WINDOW READY?
    # --------------------------------------------------------

    def ready(self):
        """
        Require approximately the complete temporal window
        before making a prediction.
        """

        if len(self.samples) < 10:
            return False

        duration = self.duration()

        return (
            duration
            >= self.window_seconds * 0.98
        )

    # --------------------------------------------------------
    # CREATE TEMPORAL SUMMARY
    # --------------------------------------------------------

    def summarize(self):
        if not self.samples:
            return None

        total_samples = len(
            self.samples
        )

        # ----------------------------------------------------
        # ONLY FRAMES WHERE A FACE WAS DETECTED
        # ----------------------------------------------------

        visible = [
            sample["features"]
            for sample in self.samples
            if sample["features"] is not None
        ]

        visible_count = len(
            visible
        )

        face_visible_ratio = (
            visible_count
            / total_samples
        )

        # ====================================================
        # NO FACE VISIBLE
        # ====================================================

        if visible_count == 0:
            return {
                "mean_yaw": 0.0,
                "std_yaw": 0.0,
                "yaw_range": 0.0,

                "mean_pitch": 0.0,
                "std_pitch": 0.0,
                "pitch_range": 0.0,

                "mean_roll": 0.0,
                "std_roll": 0.0,

                "looking_down_ratio": 0.0,

                "mean_downward_severity": 0.0,
                "max_downward_severity": 0.0,

                "forward_ratio": 0.0,
                "away_ratio": 0.0,

                "face_visible_ratio":
                    face_visible_ratio,

                "head_movement": 0.0,

                "orientation_change_rate": 0.0,

                "mean_confidence": 0.0
            }

        # ====================================================
        # NUMERICAL FEATURE ARRAYS
        # ====================================================

        yaw_values = np.array(
            [
                item["yaw"]
                for item in visible
            ],
            dtype=float
        )

        pitch_values = np.array(
            [
                item["pitch"]
                for item in visible
            ],
            dtype=float
        )

        roll_values = np.array(
            [
                item["roll"]
                for item in visible
            ],
            dtype=float
        )

        downward_values = np.array(
            [
                item["downward_severity"]
                for item in visible
            ],
            dtype=float
        )

        confidence_values = np.array(
            [
                item["confidence"]
                for item in visible
            ],
            dtype=float
        )

        # ====================================================
        # LOOKING-DOWN RATIO
        # ====================================================

        looking_down_count = sum(
            1
            for item in visible
            if item["looking_down"]
        )

        looking_down_ratio = (
            looking_down_count
            / visible_count
        )

        # ====================================================
        # FORWARD RATIO
        # ====================================================

        forward_count = sum(
            1
            for item in visible
            if item["roughly_forward"]
        )

        forward_ratio = (
            forward_count
            / visible_count
        )

        # ====================================================
        # AWAY RATIO
        # ====================================================

        away_count = sum(
            1
            for item in visible
            if item["away"]
        )

        away_ratio = (
            away_count
            / visible_count
        )

        # ====================================================
        # HEAD MOVEMENT
        #
        # Average change in yaw + pitch between consecutive
        # visible frames.
        # ====================================================

        movements = []

        orientation_changes = 0

        previous = None

        for current in visible:
            if previous is not None:
                yaw_change = abs(
                    current["yaw"]
                    -
                    previous["yaw"]
                )

                pitch_change = abs(
                    current["pitch"]
                    -
                    previous["pitch"]
                )

                movements.append(
                    yaw_change
                    +
                    pitch_change
                )

                if (
                    current["orientation"]
                    !=
                    previous["orientation"]
                ):
                    orientation_changes += 1

            previous = current

        if movements:
            head_movement = float(
                np.mean(
                    movements
                )
            )

        else:
            head_movement = 0.0

        possible_orientation_changes = max(
            1,
            visible_count - 1
        )

        orientation_change_rate = (
            orientation_changes
            /
            possible_orientation_changes
        )

        # ====================================================
        # RETURN 10-SECOND BEHAVIOURAL SUMMARY
        # ====================================================

        return {
            "mean_yaw":
                float(
                    np.mean(
                        yaw_values
                    )
                ),

            "std_yaw":
                float(
                    np.std(
                        yaw_values
                    )
                ),

            "yaw_range":
                float(
                    np.max(
                        yaw_values
                    )
                    -
                    np.min(
                        yaw_values
                    )
                ),

            "mean_pitch":
                float(
                    np.mean(
                        pitch_values
                    )
                ),

            "std_pitch":
                float(
                    np.std(
                        pitch_values
                    )
                ),

            "pitch_range":
                float(
                    np.max(
                        pitch_values
                    )
                    -
                    np.min(
                        pitch_values
                    )
                ),

            "mean_roll":
                float(
                    np.mean(
                        roll_values
                    )
                ),

            "std_roll":
                float(
                    np.std(
                        roll_values
                    )
                ),

            "looking_down_ratio":
                float(
                    looking_down_ratio
                ),

            "mean_downward_severity":
                float(
                    np.mean(
                        downward_values
                    )
                ),

            "max_downward_severity":
                float(
                    np.max(
                        downward_values
                    )
                ),

            "forward_ratio":
                float(
                    forward_ratio
                ),

            "away_ratio":
                float(
                    away_ratio
                ),

            "face_visible_ratio":
                float(
                    face_visible_ratio
                ),

            "head_movement":
                float(
                    head_movement
                ),

            "orientation_change_rate":
                float(
                    orientation_change_rate
                ),

            # Detector confidence is kept in the CSV
            # for analysis only.
            #
            # It is NOT included as an input feature
            # when training the Random Forest.
            "mean_confidence":
                float(
                    np.mean(
                        confidence_values
                    )
                )
        }