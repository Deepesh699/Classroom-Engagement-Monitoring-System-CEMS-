import csv
import os
import time

import cv2

from engagement_features import (
    extract_frame_features,
    TemporalFeatureWindow
)


# ============================================================
# CONFIGURATION
# ============================================================

YUNET_MODEL_PATH = (
    "face_detection_yunet_2023mar.onnx"
)

DATA_FOLDER = "data"

CSV_PATH = os.path.join(
    DATA_FOLDER,
    "engagement_training_v2.csv"
)

CAMERA_INDEX = 0

YUNET_CONFIDENCE = 0.75

WINDOW_SECONDS = 3.0

# Save one temporal sample every 0.5 seconds
SAVE_INTERVAL = 0.50


# ============================================================
# CSV COLUMNS
# ============================================================

CSV_COLUMNS = [
    "mean_yaw",
    "std_yaw",
    "yaw_range",

    "mean_pitch",
    "std_pitch",
    "pitch_range",

    "mean_roll",
    "std_roll",

    "looking_down_ratio",

    "mean_downward_severity",
    "max_downward_severity",

    "forward_ratio",
    "away_ratio",

    "face_visible_ratio",

    "head_movement",
    "orientation_change_rate",

    "mean_confidence",

    "label"
]


# ============================================================
# PREPARE CSV
# ============================================================

def prepare_csv():
    """
    Create the data folder and CSV file if they do not exist.
    Existing data is NOT deleted.
    New samples are appended.
    """

    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    file_exists = os.path.exists(
        CSV_PATH
    )

    if not file_exists:

        with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS
            )

            writer.writeheader()


# ============================================================
# SAVE ONE TEMPORAL SAMPLE
# ============================================================

def save_sample(
    summary,
    label
):
    """
    Save one 3-second temporal summary into the CSV.
    """

    row = dict(
        summary
    )

    row["label"] = label

    with open(
        CSV_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        writer.writerow(
            row
        )


# ============================================================
# COUNT CURRENT DATASET SAMPLES
# ============================================================

def count_samples():
    """
    Count saved rows excluding the CSV header.
    """

    if not os.path.exists(
        CSV_PATH
    ):

        return 0

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        line_count = sum(
            1
            for _ in file
        )

    return max(
        0,
        line_count - 1
    )


# ============================================================
# CHOOSE MAIN FACE
# ============================================================

def choose_largest_face(
    detections
):
    """
    Training is intentionally done with one main person.

    If several faces are visible, use the largest face.
    """

    if detections is None:

        return None

    valid_faces = []

    for detection in detections:

        confidence = float(
            detection[14]
        )

        if confidence < YUNET_CONFIDENCE:

            continue

        width = float(
            detection[2]
        )

        height = float(
            detection[3]
        )

        area = (
            width
            * height
        )

        valid_faces.append(
            (
                area,
                detection
            )
        )

    if not valid_faces:

        return None

    valid_faces.sort(
        key=lambda item:
            item[0],
        reverse=True
    )

    return valid_faces[0][1]


# ============================================================
# DRAW STATUS PANEL
# ============================================================

def draw_panel(
    frame,
    current_label,
    orientation,
    sample_count,
    window_ready,
    current_features
):

    height, width = (
        frame.shape[:2]
    )

    panel_width = min(
        700,
        width - 20
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (panel_width, 250),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # CURRENT LABEL
    # --------------------------------------------------------

    if current_label is None:

        label_text = "PAUSED"

    else:

        label_text = current_label

    cv2.putText(
        frame,
        f"Training Label: {label_text}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # ORIENTATION
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Orientation: {orientation}",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # DATASET COUNT
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Saved samples: {sample_count}",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # TEMPORAL WINDOW STATUS
    # --------------------------------------------------------

    if window_ready:

        window_text = (
            "Temporal window: READY"
        )

    else:

        window_text = (
            "Temporal window: collecting..."
        )

    cv2.putText(
        frame,
        window_text,
        (25, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # LIVE BEHAVIOURAL SIGNALS
    # --------------------------------------------------------

    if current_features is not None:

        yaw = current_features[
            "yaw"
        ]

        pitch = current_features[
            "pitch"
        ]

        downward = current_features[
            "downward_severity"
        ]

        away = current_features[
            "away"
        ]

        cv2.putText(
            frame,
            (
                f"Yaw: {yaw:.3f} | "
                f"Pitch: {pitch:.3f}"
            ),
            (25, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                f"Down severity: {downward:.2f} | "
                f"Away: {away}"
            ),
            (25, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


# ============================================================
# PRINT CONTROLS
# ============================================================

def print_controls():

    print()

    print(
        "======================================"
    )

    print(
        " CEMS ENGAGEMENT DATA COLLECTION V2"
    )

    print(
        "======================================"
    )

    print()

    print(
        "1 = Engaged"
    )

    print(
        "2 = Neutral"
    )

    print(
        "3 = Low Engagement"
    )

    print(
        "0 = Pause recording"
    )

    print(
        "Q = Quit"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Select the label BEFORE performing "
        "the behaviour."
    )

    print()

    print(
        "Try to perform natural behaviours, "
        "not exaggerated poses."
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK YUNET MODEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREPARE DATASET
    # --------------------------------------------------------

    prepare_csv()

    sample_count = (
        count_samples()
    )

    # --------------------------------------------------------
    # CREATE YUNET
    # --------------------------------------------------------

    detector = (
        cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (
                320,
                320
            ),
            YUNET_CONFIDENCE,
            0.3,
            5000
        )
    )

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print()

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

    # --------------------------------------------------------
    # TEMPORAL WINDOW
    # --------------------------------------------------------

    temporal_window = (
        TemporalFeatureWindow(
            window_seconds=
            WINDOW_SECONDS
        )
    )

    current_label = None

    last_save_time = 0.0

    print_controls()

    print(
        f"Dataset: {CSV_PATH}"
    )

    print(
        f"Existing samples: {sample_count}"
    )

    print()

    # ========================================================
    # MAIN CAMERA LOOP
    # ========================================================

    while True:

        success, frame = (
            camera.read()
        )

        if not success:

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

        # ----------------------------------------------------
        # YUNET DETECTION
        # ----------------------------------------------------

        _, detections = (
            detector.detect(
                frame
            )
        )

        detection = (
            choose_largest_face(
                detections
            )
        )

        orientation = (
            "Face Not Visible"
        )

        current_features = None

        # ====================================================
        # FACE FOUND
        # ====================================================

        if detection is not None:

            current_features = (
                extract_frame_features(
                    detection
                )
            )

            temporal_window.add(
                current_features
            )

            orientation = (
                current_features[
                    "orientation"
                ]
            )

            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            x = int(
                detection[0]
            )

            y = int(
                detection[1]
            )

            w = int(
                detection[2]
            )

            h = int(
                detection[3]
            )

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
                (
                    0,
                    255,
                    0
                ),
                2
            )

            cv2.putText(
                frame,
                orientation,
                (
                    x,
                    max(
                        25,
                        y - 10
                    )
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

        # ====================================================
        # FACE NOT FOUND
        # ====================================================

        else:

            # None means the temporal window knows
            # the face was not visible during this frame.
            temporal_window.add(
                None
            )

        # ====================================================
        # SAVE TEMPORAL SAMPLE
        # ====================================================

        now = time.monotonic()

        window_ready = (
            temporal_window.ready()
        )

        if (
            current_label is not None
            and
            window_ready
            and
            now - last_save_time
            >= SAVE_INTERVAL
        ):

            summary = (
                temporal_window.summarize()
            )

            if summary is not None:

                save_sample(
                    summary,
                    current_label
                )

                sample_count += 1

                last_save_time = now

        # ====================================================
        # DRAW UI
        # ====================================================

        draw_panel(
            frame,
            current_label,
            orientation,
            sample_count,
            window_ready,
            current_features
        )

        cv2.putText(
            frame,
            (
                "1 Engaged | "
                "2 Neutral | "
                "3 Low | "
                "0 Pause | "
                "Q Quit"
            ),
            (
                20,
                height - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "CEMS - Engagement Training Data V2",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # ====================================================
        # LABEL CONTROLS
        # ====================================================

        if key == ord(
            "1"
        ):

            current_label = (
                "Engaged"
            )

            temporal_window.clear()

            last_save_time = (
                time.monotonic()
            )

            print()

            print(
                "Recording: ENGAGED"
            )

        elif key == ord(
            "2"
        ):

            current_label = (
                "Neutral"
            )

            temporal_window.clear()

            last_save_time = (
                time.monotonic()
            )

            print()

            print(
                "Recording: NEUTRAL"
            )

        elif key == ord(
            "3"
        ):

            current_label = (
                "Low Engagement"
            )

            temporal_window.clear()

            last_save_time = (
                time.monotonic()
            )

            print()

            print(
                "Recording: LOW ENGAGEMENT"
            )

        elif key == ord(
            "0"
        ):

            current_label = None

            temporal_window.clear()

            print()

            print(
                "Recording paused."
            )

        elif key == ord(
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
        "======================================"
    )

    print(
        "DATA COLLECTION FINISHED"
    )

    print(
        "======================================"
    )

    print(
        f"Dataset: {CSV_PATH}"
    )

    print(
        f"Total samples: {sample_count}"
    )

    print()


if __name__ == "__main__":

    main()