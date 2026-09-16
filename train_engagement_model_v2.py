import os
import sys

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.model_selection import (
    train_test_split
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "engagement_training_v2.csv"
)

MODEL_PATH = (
    "engagement_model_v2.joblib"
)

RANDOM_STATE = 42


# ============================================================
# FEATURES
#
# IMPORTANT:
# mean_confidence is intentionally NOT included.
#
# We want the model to learn behavioural signals,
# not simply detector confidence.
# ============================================================

FEATURE_COLUMNS = [
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

    "orientation_change_rate"
]


TARGET_COLUMN = "label"


# ============================================================
# EXPECTED LABELS
# ============================================================

EXPECTED_LABELS = [
    "Engaged",
    "Neutral",
    "Low Engagement"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "======================================"
    )

    print(
        " CEMS ENGAGEMENT MODEL V2 TRAINING"
    )

    print(
        "======================================"
    )

    print()

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not os.path.exists(
        DATA_PATH
    ):

        print(
            f"ERROR: Dataset not found: "
            f"{DATA_PATH}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset loaded: {DATA_PATH}"
    )

    print(
        f"Total rows: {len(data)}"
    )

    print()

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns

        if column
        not in data.columns
    ]

    if missing_columns:

        print(
            "ERROR: Dataset is missing "
            "required columns:"
        )

        for column in missing_columns:

            print(
                f" - {column}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # DROP INVALID ROWS
    # --------------------------------------------------------

    data = data.dropna(
        subset=required_columns
    )

    print(
        f"Valid rows: {len(data)}"
    )

    print()

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    class_counts = (
        data[
            TARGET_COLUMN
        ]
        .value_counts()
    )

    print(
        "Samples per class:"
    )

    print()

    print(
        class_counts
    )

    print()

    # --------------------------------------------------------
    # CHECK CLASSES
    # --------------------------------------------------------

    existing_labels = set(
        data[
            TARGET_COLUMN
        ].unique()
    )

    missing_labels = [
        label
        for label in EXPECTED_LABELS

        if label
        not in existing_labels
    ]

    if missing_labels:

        print(
            "WARNING:"
        )

        print(
            "The following expected classes "
            "are missing:"
        )

        for label in missing_labels:

            print(
                f" - {label}"
            )

        print()

    if len(existing_labels) < 2:

        print(
            "ERROR: At least two classes "
            "are required."
        )

        sys.exit(1)

    # ========================================================
    # INPUT AND TARGET
    # ========================================================

    X = data[
        FEATURE_COLUMNS
    ]

    y = data[
        TARGET_COLUMN
    ]

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,

            test_size=0.25,

            random_state=
            RANDOM_STATE,

            stratify=y
        )
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    print()

    # ========================================================
    # RANDOM FOREST V2
    # ========================================================

    model = RandomForestClassifier(

        n_estimators=500,

        max_depth=None,

        min_samples_split=4,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=
        RANDOM_STATE,

        n_jobs=-1
    )

    print(
        "Training Random Forest V2..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    print()

    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = model.predict(
        X_test
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    macro_f1 = f1_score(
        y_test,
        predictions,

        average="macro",

        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,

        average="weighted",

        zero_division=0
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "======================================"
    )

    print(
        " PRELIMINARY V2 RESULTS"
    )

    print(
        "======================================"
    )

    print()

    print(
        f"Accuracy: "
        f"{accuracy:.3f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.3f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.3f}"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1:.3f}"
    )

    print()

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print(
        "Classification Report:"
    )

    print()

    print(
        classification_report(
            y_test,
            predictions,

            labels=
            EXPECTED_LABELS,

            zero_division=0
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_test,
        predictions,

        labels=
        EXPECTED_LABELS
    )

    matrix_df = pd.DataFrame(
        matrix,

        index=[
            f"Actual {label}"
            for label in EXPECTED_LABELS
        ],

        columns=[
            f"Pred {label}"
            for label in EXPECTED_LABELS
        ]
    )

    print(
        "======================================"
    )

    print(
        " CONFUSION MATRIX"
    )

    print(
        "======================================"
    )

    print()

    print(
        matrix_df
    )

    print()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = (
        pd.DataFrame(
            {
                "feature":
                    FEATURE_COLUMNS,

                "importance":
                    model.feature_importances_
            }
        )
    )

    feature_importance = (
        feature_importance.sort_values(
            by="importance",
            ascending=False
        )
    )

    print(
        "======================================"
    )

    print(
        " FEATURE IMPORTANCE"
    )

    print(
        "======================================"
    )

    print()

    print(
        feature_importance.to_string(
            index=False
        )
    )

    print()

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_bundle = {

        "model":
            model,

        "features":
            FEATURE_COLUMNS,

        "classes":
            list(
                model.classes_
            ),

        "window_seconds":
            3.0,

        "version":
            2,

        "uses_detector_confidence":
            False
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    print(
        "======================================"
    )

    print(
        " MODEL SAVED"
    )

    print(
        "======================================"
    )

    print()

    print(
        f"Saved model: "
        f"{MODEL_PATH}"
    )

    print()

    print(
        "Model uses behavioural "
        "features only."
    )

    print(
        "YuNet mean confidence was "
        "excluded from training."
    )

    print()

    # ========================================================
    # IMPORTANT EVALUATION WARNING
    # ========================================================

    print(
        "======================================"
    )

    print(
        " EVALUATION NOTE"
    )

    print(
        "======================================"
    )

    print()

    print(
        "These results are still "
        "PRELIMINARY."
    )

    print()

    print(
        "The dataset contains overlapping "
        "3-second windows."
    )

    print(
        "Therefore similar samples may "
        "appear in both training and testing."
    )

    print()

    print(
        "Final evaluation should use "
        "completely separate recording "
        "sessions or participants."
    )

    print()


if __name__ == "__main__":

    main()