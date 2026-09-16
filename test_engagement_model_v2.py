import os

import cv2
import joblib
import pandas as pd

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

ENGAGEMENT_MODEL_PATH = (
    "engagement_model_v2.joblib"
)

YUNET_CONFIDENCE = 0.75

WINDOW_SECONDS = 3.0

CAMERA_INDEX = 0


# ============================================================
# LOAD MODEL
# ============================================================

def load_engagement_model():

    if not os.path.exists(
        ENGAGEMENT_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found: "
            f"{ENGAGEMENT_MODEL_PATH}"
        )

    bundle = joblib.load(
        ENGAGEMENT_MODEL_PATH
    )

    return (
        bundle["model"],
        bundle["features"]
    )


# ============================================================
# CHOOSE MAIN FACE
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

        if (
            confidence
            < YUNET_CONFIDENCE
        ):

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

    return (
        valid_faces[0][1]
    )


# ============================================================
# CALCULATE ENGAGEMENT SCORE
# ============================================================

def calculate_engagement_score(
    model,
    probabilities
):
    """
    Convert class probabilities into a display score.

    This is a derived engagement score,
    not a separately trained continuous target.
    """

    probability_map = {}

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):

        probability_map[
            class_name
        ] = float(
            probability
        )

    engaged_probability = (
        probability_map.get(
            "Engaged",
            0.0
        )
    )

    neutral_probability = (
        probability_map.get(
            "Neutral",
            0.0
        )
    )

    low_probability = (
        probability_map.get(
            "Low Engagement",
            0.0
        )
    )

    # Weighted class score:
    #
    # Engaged        ~ 90
    # Neutral        ~ 60
    # Low Engagement ~ 30
    #
    # Probabilities allow smooth transitions.

    score = (
        engaged_probability
        * 90.0

        +

        neutral_probability
        * 60.0

        +

        low_probability
        * 30.0
    )

    score = round(
        score
    )

    score = max(
        20,
        min(
            95,
            score
        )
    )

    return score


# ============================================================
# DRAW PANEL
# ============================================================

def draw_panel(
    frame,
    status,
    score,
    model_confidence,
    orientation,
    window_ready,
    current_features
):

    cv2.rectangle(
        frame,
        (10, 10),
        (680, 265),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"ML Engagement: {status}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Engagement Score: {score}%",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        (
            "Model confidence: "
            f"{model_confidence:.1%}"
        ),
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Orientation: {orientation}",
        (25, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    if window_ready:

        window_text = (
            "3-second behaviour window: READY"
        )

    else:

        window_text = (
            "3-second behaviour window: collecting..."
        )

    cv2.putText(
        frame,
        window_text,
        (25, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

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

        cv2.putText(
            frame,
            (
                f"Yaw: {yaw:.3f} | "
                f"Pitch: {pitch:.3f}"
            ),
            (25, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                f"Down: {downward:.2f} | "
                f"Away: {away}"
            ),
            (360, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK YUNET
    # --------------------------------------------------------

    if not os.path.exists(
        YUNET_MODEL_PATH
    ):

        print(
            "ERROR: YuNet model not found."
        )

        return

    # --------------------------------------------------------
    # LOAD ENGAGEMENT MODEL
    # --------------------------------------------------------

    try:

        (
            model,
            feature_columns
        ) = load_engagement_model()

    except Exception as error:

        print(
            f"ERROR loading model: "
            f"{error}"
        )

        return

    print()
    print(
        "======================================"
    )
    print(
        " CEMS LIVE ML ENGAGEMENT TEST"
    )
    print(
        "======================================"
    )
    print()

    print(
        "Model loaded:"
    )

    print(
        ENGAGEMENT_MODEL_PATH
    )

    print()

    print(
        "Model classes:"
    )

    print(
        model.classes_
    )

    print()

    # --------------------------------------------------------
    # YUNET
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
    # CAMERA
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
    # TEMPORAL WINDOW
    # --------------------------------------------------------

    temporal_window = (
        TemporalFeatureWindow(
            window_seconds=
            WINDOW_SECONDS
        )
    )

    status = (
        "Collecting..."
    )

    score = 0

    model_confidence = 0.0

    orientation = (
        "Unknown"
    )

    print(
        "Move naturally."
    )

    print(
        "Try forward, looking down, "
        "sideways and mixed behaviour."
    )

    print()

    print(
        "Press Q to quit."
    )

    print()

    # ========================================================
    # LIVE LOOP
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
                (x, y),
                (
                    x + w,
                    y + h
                ),
                (0, 255, 0),
                2
            )

        # ====================================================
        # FACE NOT FOUND
        # ====================================================

        else:

            temporal_window.add(
                None
            )

            orientation = (
                "Face Not Visible"
            )

        # ====================================================
        # ML PREDICTION
        # ====================================================

        window_ready = (
            temporal_window.ready()
        )

        if window_ready:

            summary = (
                temporal_window.summarize()
            )

            if summary is not None:

                # --------------------------------------------
                # EXACT FEATURES EXPECTED BY TRAINED MODEL
                # --------------------------------------------

                model_input = {
                    feature:
                        summary[
                            feature
                        ]

                    for feature
                    in feature_columns
                }

                input_df = (
                    pd.DataFrame(
                        [
                            model_input
                        ],
                        columns=
                        feature_columns
                    )
                )

                # --------------------------------------------
                # CLASS
                # --------------------------------------------

                prediction = (
                    model.predict(
                        input_df
                    )[0]
                )

                # --------------------------------------------
                # PROBABILITIES
                # --------------------------------------------

                probabilities = (
                    model.predict_proba(
                        input_df
                    )[0]
                )

                status = (
                    str(
                        prediction
                    )
                )

                model_confidence = (
                    float(
                        max(
                            probabilities
                        )
                    )
                )

                score = (
                    calculate_engagement_score(
                        model,
                        probabilities
                    )
                )

        # ====================================================
        # STATUS COLOUR
        # ====================================================

        if status == "Engaged":

            status_colour = (
                0,
                255,
                0
            )

        elif status == "Neutral":

            status_colour = (
                0,
                255,
                255
            )

        elif (
            status
            == "Low Engagement"
        ):

            status_colour = (
                0,
                0,
                255
            )

        else:

            status_colour = (
                255,
                255,
                255
            )

        # ====================================================
        # MAIN RESULT TEXT
        # ====================================================

        cv2.putText(
            frame,
            (
                f"{status} | "
                f"{score}%"
            ),
            (
                20,
                height - 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            status_colour,
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        draw_panel(
            frame,
            status,
            score,
            model_confidence,
            orientation,
            window_ready,
            current_features
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "CEMS - Live ML Engagement V2",
            frame
        )

        key = (
            cv2.waitKey(1)
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
        "Live engagement test finished."
    )


if __name__ == "__main__":

    main()