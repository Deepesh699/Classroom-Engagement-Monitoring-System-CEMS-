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


# ============================================================
# TEMPORAL SETTINGS
# ============================================================

# Engagement is analysed over the most recent 10 seconds.
WINDOW_SECONDS = 10.0

# Save approximately one independent temporal sample every
# 10 seconds.
#
# This is intentionally not 0.5 seconds because extremely
# overlapping samples can make evaluation look unrealistically
# strong.
SAVE_INTERVAL = 10.0


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
# CREATE DATASET IF NEEDED
# ============================================================

def prepare_csv():
    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    if not os.path.exists(
        CSV_PATH
    ):
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
    row = dict(
        summary
    )

    row["label"] = (
        label
    )

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
# COUNT EXISTING SAMPLES
# ============================================================

def count_samples():
    if not os.path.exists(
        CSV_PATH
    ):
        return 0

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        lines = sum(
            1
            for _ in file
        )

    return max(
        0,
        lines - 1
    )


# ============================================================
# SELECT MAIN FACE
#
# For training data collection we use the largest detected
# face so that one person is labelled at a time.
# ============================================================

def choose_largest_face(
    detections
):
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
            *
            height
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

    return (
        valid_faces[0][1]
    )


# ============================================================
# DRAW INFORMATION PANEL
# ============================================================

def draw_panel(
    frame,
    current_label,
    orientation,
    sample_count,
    window_ready,
    window_duration,
    current_features
):
    height, width = (
        frame.shape[:2]
    )

    panel_width = min(
        760,
        width - 20
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (panel_width, 315),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # CURRENT TRAINING LABEL
    # --------------------------------------------------------

    if current_label is None:
        label_text = "PAUSED"

    else:
        label_text = (
            current_label
        )

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
    # CURRENT ORIENTATION
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
    # SAVED SAMPLES
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Saved Samples: {sample_count}",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # WINDOW PROGRESS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        (
            f"Behaviour Window: "
            f"{window_duration:.1f} / "
            f"{WINDOW_SECONDS:.0f} sec"
        ),
        (25, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    if window_ready:
        window_text = (
            "Temporal Window: READY"
        )

    else:
        window_text = (
            "Temporal Window: COLLECTING"
        )

    cv2.putText(
        frame,
        window_text,
        (25, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # CURRENT FRAME FEATURES
    # --------------------------------------------------------

    if current_features is not None:
        yaw = (
            current_features[
                "yaw"
            ]
        )

        pitch = (
            current_features[
                "pitch"
            ]
        )

        downward = (
            current_features[
                "downward_severity"
            ]
        )

        away = (
            current_features[
                "away"
            ]
        )

        confidence = (
            current_features[
                "confidence"
            ]
        )

        cv2.putText(
            frame,
            (
                f"Yaw: {yaw:.3f} | "
                f"Pitch: {pitch:.3f}"
            ),
            (25, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        away_text = (
            "Yes"
            if away
            else "No"
        )

        cv2.putText(
            frame,
            (
                f"Downward Level: "
                f"{downward * 100:.0f}% | "
                f"Looking Away: {away_text}"
            ),
            (25, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                f"YuNet Face Confidence: "
                f"{confidence * 100:.1f}%"
            ),
            (25, 290),
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
    print("==========================================")
    print(" CEMS ENGAGEMENT DATA COLLECTION V2")
    print(" 10-SECOND TEMPORAL WINDOW")
    print("==========================================")
    print()
    print("1 = Engaged")
    print("2 = Neutral")
    print("3 = Low Engagement")
    print("0 = Pause")
    print("Q = Quit")
    print()
    print(
        "Perform each behaviour naturally for "
        "at least 30-60 seconds."
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

    prepare_csv()

    sample_count = (
        count_samples()
    )

    # --------------------------------------------------------
    # CREATE YUNET DETECTOR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():
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
    # CREATE 10-SECOND TEMPORAL WINDOW
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
    # CAMERA LOOP
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

        # ----------------------------------------------------
        # UPDATE YUNET INPUT SIZE
        # ----------------------------------------------------

        detector.setInputSize(
            (
                width,
                height
            )
        )

        # ----------------------------------------------------
        # FACE DETECTION
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
        # FACE DETECTED
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
            # DRAW FACE BOX
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
        # FACE NOT DETECTED
        # ====================================================

        else:
            temporal_window.add(
                None
            )

        # ====================================================
        # WINDOW STATUS
        # ====================================================

        now = time.monotonic()

        window_ready = (
            temporal_window.ready()
        )

        window_duration = (
            temporal_window.duration()
        )

        # ====================================================
        # SAVE TRAINING SAMPLE
        # ====================================================

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

                last_save_time = (
                    now
                )

                print(
                    f"Saved sample "
                    f"#{sample_count} "
                    f"-> {current_label}"
                )

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        draw_panel(
            frame,
            current_label,
            orientation,
            sample_count,
            window_ready,
            window_duration,
            current_features
        )

        # ----------------------------------------------------
        # CONTROLS AT BOTTOM
        # ----------------------------------------------------

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
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # SHOW WINDOW
        # ----------------------------------------------------

        cv2.imshow(
            (
                "CEMS - Engagement Data Collection V2 "
                "(10 Seconds)"
            ),
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # ====================================================
        # LABEL CONTROLS
        # ====================================================

        if key == ord("1"):

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

        elif key == ord("2"):

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

        elif key == ord("3"):

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

        elif key == ord("0"):

            current_label = None

            temporal_window.clear()

            print()
            print(
                "Recording paused."
            )

        elif key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("==========================================")
    print(" DATA COLLECTION FINISHED")
    print("==========================================")
    print(
        f"Dataset: {CSV_PATH}"
    )
    print(
        f"Total samples: {sample_count}"
    )
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()