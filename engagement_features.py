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

    YuNet:
        0-3   = face bounding box
        4-5   = right eye
        6-7   = left eye
        8-9   = nose
        10-11 = right mouth
        12-13 = left mouth
        14    = confidence
    """

    x = float(detection[0])
    y = float(detection[1])
    w = float(detection[2])
    h = float(detection[3])

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

    confidence = float(detection[14])

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
    # ========================================================

    yaw = (
        nose[0] - eye_mid_x
    ) / eye_distance

    # ========================================================
    # PITCH
    # ========================================================

    pitch = (
        nose[1] - eye_mid_y
    ) / face_vertical

    # ========================================================
    # ROLL
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
        eye_distance
        / face_width
    )

    # ========================================================
    # BEHAVIOURAL SIGNALS
    # ========================================================

    # Looking down threshold.
    looking_down = (
        pitch > 0.72
    )

    # Continuous downward value instead of only True / False.
    #
    # 0.0 = not meaningfully downward
    # 1.0 = strongly downward
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

    roughly_forward = (
        abs(yaw) <= 0.32
        and
        0.32 <= pitch <= 0.72
    )

    # "Away" is deliberately broader than only left/right.
    away = (
        abs(yaw) > 0.32
        or
        pitch > 0.72
        or
        pitch < 0.32
    )

    # ========================================================
    # ORIENTATION LABEL
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

    return {
        "yaw":
            yaw,

        "pitch":
            pitch,

        "roll":
            roll,

        "confidence":
            confidence,

        "eye_distance_ratio":
            eye_distance_ratio,

        "looking_down":
            looking_down,

        "downward_severity":
            downward_severity,

        "roughly_forward":
            roughly_forward,

        "away":
            away,

        "orientation":
            orientation
    }


# ============================================================
# TEMPORAL FEATURE WINDOW
# ============================================================

class TemporalFeatureWindow:
    """
    Stores several seconds of behaviour and converts those
    frames into one temporal ML sample.
    """

    def __init__(
        self,
        window_seconds=3.0
    ):

        self.window_seconds = (
            window_seconds
        )

        self.samples = deque()

    # --------------------------------------------------------
    # RESET WINDOW
    # --------------------------------------------------------

    def clear(self):

        self.samples.clear()

    # --------------------------------------------------------
    # ADD CURRENT FRAME
    # --------------------------------------------------------

    def add(
        self,
        features
    ):

        now = time.monotonic()

        self.samples.append(
            {
                "time":
                    now,

                "features":
                    features
            }
        )

        self._remove_old(
            now
        )

    # --------------------------------------------------------
    # REMOVE OLD FRAMES
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
            self.samples[0]["time"]
            < cutoff
        ):

            self.samples.popleft()

    # --------------------------------------------------------
    # IS WINDOW READY?
    # --------------------------------------------------------

    def ready(self):

        if len(self.samples) < 10:

            return False

        duration = (
            self.samples[-1]["time"]
            -
            self.samples[0]["time"]
        )

        return (
            duration
            >= self.window_seconds
            * 0.80
        )

    # --------------------------------------------------------
    # TEMPORAL SUMMARY
    # --------------------------------------------------------

    def summarize(self):

        if not self.samples:

            return None

        total_samples = len(
            self.samples
        )

        visible = [
            sample["features"]

            for sample
            in self.samples

            if sample["features"]
            is not None
        ]

        visible_count = len(
            visible
        )

        face_visible_ratio = (
            visible_count
            / total_samples
        )

        # ====================================================
        # NO VISIBLE FACE
        # ========================================================

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
        # ARRAYS
        # ========================================================

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
                item[
                    "downward_severity"
                ]
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
        # RATIOS
        # ========================================================

        looking_down_ratio = (
            sum(
                1
                for item in visible
                if item[
                    "looking_down"
                ]
            )
            / visible_count
        )

        forward_ratio = (
            sum(
                1
                for item in visible
                if item[
                    "roughly_forward"
                ]
            )
            / visible_count
        )

        away_ratio = (
            sum(
                1
                for item in visible
                if item[
                    "away"
                ]
            )
            / visible_count
        )

        # ====================================================
        # MOVEMENT + ORIENTATION CHANGES
        # ========================================================

        movements = []

        orientation_changes = 0

        previous = None

        for current in visible:

            if previous is not None:

                yaw_change = abs(
                    current["yaw"]
                    - previous["yaw"]
                )

                pitch_change = abs(
                    current["pitch"]
                    - previous["pitch"]
                )

                movements.append(
                    yaw_change
                    + pitch_change
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

        possible_changes = max(
            1,
            visible_count - 1
        )

        orientation_change_rate = (
            orientation_changes
            / possible_changes
        )

        # ====================================================
        # RETURN TEMPORAL FEATURES
        # ========================================================

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

            # Keep for analysis only.
            # We may NOT use this for training later.
            "mean_confidence":
                float(
                    np.mean(
                        confidence_values
                    )
                )
        }