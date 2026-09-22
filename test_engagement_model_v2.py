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

CAMERA_INDEX = 0


# ============================================================
# LOAD TRAINED ENGAGEMENT MODEL
# ============================================================

def load_engagement_model():
    """
    Load the trained Random Forest model bundle.

    The temporal window duration is read directly from the
    saved model so the live test always uses the same window
    duration that was used during training.
    """

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

    model = (
        bundle[
            "model"
        ]
    )

    # --------------------------------------------------------
    # USE ONE CPU THREAD DURING LIVE PREDICTION
    #
    # This does not modify the trained Random Forest.
    # It simply avoids unnecessary parallel-processing
    # warnings during live prediction.
    # --------------------------------------------------------

    model.n_jobs = 1

    feature_columns = (
        bundle[
            "features"
        ]
    )

    # --------------------------------------------------------
    # READ WINDOW FROM TRAINED MODEL
    #
    # Current model should contain:
    #
    # "window_seconds": 10.0
    #
    # 10.0 is also used as the fallback.
    # --------------------------------------------------------

    window_seconds = float(
        bundle.get(
            "window_seconds",
            10.0
        )
    )

    return (
        model,
        feature_columns,
        window_seconds
    )


# ============================================================
# SELECT MAIN FACE
# ============================================================

def choose_largest_face(
    detections
):
    """
    For this single-person live ML test, select the largest
    valid detected face.

    Multi-student processing will later happen inside the
    tracking pipeline using one temporal history per track_id.
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
# CALCULATE DISPLAY ENGAGEMENT SCORE
# ============================================================

def calculate_engagement_score(
    model,
    probabilities
):
    """
    Convert Random Forest class probabilities into the
    display engagement score.

    IMPORTANT:

    This is a derived display score.

    The Random Forest is trained to classify:

        Engaged
        Neutral
        Low Engagement

    It is not trained directly to predict a continuous
    percentage score.
    """

    probability_map = {}

    for (
        class_name,
        probability
    ) in zip(
        model.classes_,
        probabilities
    ):

        probability_map[
            str(
                class_name
            )
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

    # --------------------------------------------------------
    # WEIGHTED DISPLAY SCORE
    # --------------------------------------------------------

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

    # Keep display score in a sensible visible range.
    return max(
        20,
        min(
            95,
            score
        )
    )


# ============================================================
# DRAW INFORMATION PANEL
# ============================================================

def draw_panel(
    frame,
    status,
    score,
    model_confidence,
    orientation,
    window_ready,
    window_duration,
    window_seconds,
    current_features
):
    """
    Draw live engagement information on the video frame.
    """

    # --------------------------------------------------------
    # PANEL BACKGROUND
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (780, 350),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # ML ENGAGEMENT CLASS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DERIVED ENGAGEMENT SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RANDOM FOREST CONFIDENCE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        (
            "Model Confidence: "
            f"{model_confidence:.1%}"
        ),
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # CURRENT ORIENTATION
    # --------------------------------------------------------

    cv2.putText(
        frame,
        (
            f"Orientation: "
            f"{orientation}"
        ),
        (25, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
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
            f"Behaviour History: "
            f"{window_duration:.1f} / "
            f"{window_seconds:.0f} sec"
        ),
        (25, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # WINDOW READY STATUS
    #
    # IMPORTANT:
    # Duration is generated dynamically from window_seconds.
    # Nothing is hardcoded to 10 or 15 here.
    # --------------------------------------------------------

    if window_ready:

        readiness = (
            f"{window_seconds:.0f}-second "
            f"window: READY"
        )

    else:

        readiness = (
            f"{window_seconds:.0f}-second "
            f"window: COLLECTING"
        )

    cv2.putText(
        frame,
        readiness,
        (25, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # CURRENT FRAME BEHAVIOURAL FEATURES
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

        face_confidence = (
            current_features[
                "confidence"
            ]
        )

        # ----------------------------------------------------
        # YAW / PITCH
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"Yaw: {yaw:.3f} | "
                f"Pitch: {pitch:.3f}"
            ),
            (25, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # LOOKING AWAY
        # ----------------------------------------------------

        if away:
            away_text = "Yes"

        else:
            away_text = "No"

        cv2.putText(
            frame,
            (
                f"Downward Level: "
                f"{downward * 100:.0f}% | "
                f"Looking Away: {away_text}"
            ),
            (25, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # YUNET DETECTOR CONFIDENCE
        #
        # This is NOT the Random Forest engagement confidence.
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"YuNet Face Confidence: "
                f"{face_confidence * 100:.1f}%"
            ),
            (25, 325),
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
    # CHECK YUNET FILE
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
    # LOAD RANDOM FOREST MODEL
    # --------------------------------------------------------

    try:

        (
            model,
            feature_columns,
            window_seconds
        ) = load_engagement_model()

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
    # STARTUP INFORMATION
    # ========================================================

    print()

    print(
        "=========================================="
    )

    print(
        " CEMS LIVE ML ENGAGEMENT V2"
    )

    print(
        f" {window_seconds:.0f}-SECOND TEMPORAL WINDOW"
    )

    print(
        "=========================================="
    )

    print()

    print(
        f"Model: "
        f"{ENGAGEMENT_MODEL_PATH}"
    )

    print(
        f"Temporal Window: "
        f"{window_seconds:.0f} seconds"
    )

    print()

    print(
        "Model Classes:"
    )

    print(
        model.classes_
    )

    print()

    print(
        "Live prediction CPU threads: 1"
    )

    print()

    # ========================================================
    # CREATE YUNET DETECTOR
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
    # OPEN CAMERA
    # ========================================================

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

    # ========================================================
    # CREATE TEMPORAL WINDOW
    #
    # This gets its duration from the saved trained model.
    # ========================================================

    temporal_window = (
        TemporalFeatureWindow(
            window_seconds=
            window_seconds
        )
    )

    # ========================================================
    # INITIAL DISPLAY VALUES
    # ========================================================

    status = (
        "Collecting..."
    )

    score = 0

    model_confidence = 0.0

    orientation = (
        "Unknown"
    )

    print(
        f"Collecting approximately "
        f"{window_seconds:.0f} seconds "
        f"before the first prediction."
    )

    print()

    print(
        "Press Q to quit."
    )

    print()

    # ========================================================
    # LIVE CAMERA LOOP
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
        # DETECT FACES
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
        # FACE DETECTED
        # ====================================================

        if detection is not None:

            # ------------------------------------------------
            # EXTRACT CURRENT-FRAME BEHAVIOURAL FEATURES
            # ------------------------------------------------

            current_features = (
                extract_frame_features(
                    detection
                )
            )

            # ------------------------------------------------
            # ADD CURRENT FRAME TO TEMPORAL HISTORY
            # ------------------------------------------------

            temporal_window.add(
                current_features
            )

            # ------------------------------------------------
            # CURRENT HEAD ORIENTATION
            # ------------------------------------------------

            orientation = (
                current_features[
                    "orientation"
                ]
            )

            # ------------------------------------------------
            # FACE BOUNDING BOX
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

            # ------------------------------------------------
            # ORIENTATION LABEL ABOVE FACE
            # ------------------------------------------------

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

            # Store missing visibility in temporal history.
            temporal_window.add(
                None
            )

            orientation = (
                "Face Not Visible"
            )

        # ====================================================
        # TEMPORAL WINDOW STATUS
        # ====================================================

        window_ready = (
            temporal_window.ready()
        )

        window_duration = (
            temporal_window.duration()
        )

        # ====================================================
        # RANDOM FOREST ENGAGEMENT PREDICTION
        # ====================================================

        if window_ready:

            summary = (
                temporal_window.summarize()
            )

            if summary is not None:

                # --------------------------------------------
                # BUILD EXACT FEATURE VECTOR USED BY TRAINING
                # --------------------------------------------

                model_input = {
                    feature:
                        summary[
                            feature
                        ]

                    for feature
                    in feature_columns
                }

                input_df = pd.DataFrame(
                    [
                        model_input
                    ],
                    columns=
                    feature_columns
                )

                # --------------------------------------------
                # PREDICT ENGAGEMENT CLASS
                # --------------------------------------------

                prediction = (
                    model.predict(
                        input_df
                    )[0]
                )

                # --------------------------------------------
                # GET RANDOM FOREST CLASS PROBABILITIES
                # --------------------------------------------

                probabilities = (
                    model.predict_proba(
                        input_df
                    )[0]
                )

                # --------------------------------------------
                # ENGAGEMENT CLASS
                # --------------------------------------------

                status = str(
                    prediction
                )

                # --------------------------------------------
                # MODEL CONFIDENCE
                #
                # Highest probability among:
                #
                # Engaged
                # Neutral
                # Low Engagement
                # --------------------------------------------

                model_confidence = float(
                    max(
                        probabilities
                    )
                )

                # --------------------------------------------
                # DERIVED DISPLAY SCORE
                # --------------------------------------------

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

        elif status == "Low Engagement":

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
        # MAIN RESULT AT BOTTOM OF SCREEN
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
        # DRAW INFORMATION PANEL
        # ====================================================

        draw_panel(
            frame,
            status,
            score,
            model_confidence,
            orientation,
            window_ready,
            window_duration,
            window_seconds,
            current_features
        )

        # ====================================================
        # WINDOW TITLE
        #
        # Also generated from the real model duration.
        # ====================================================

        window_title = (
            "CEMS - ML Engagement V2 "
            f"({window_seconds:.0f} Seconds)"
        )

        cv2.imshow(
            window_title,
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

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

    print()


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()