import os
import sys

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "engagement_training.csv"
)

MODEL_PATH = "engagement_model.joblib"

RANDOM_STATE = 42


# ============================================================
# FEATURES USED BY THE MODEL
# ============================================================

FEATURE_COLUMNS = [
    "mean_yaw",
    "std_yaw",

    "mean_pitch",
    "std_pitch",

    "mean_roll",
    "std_roll",

    "looking_down_ratio",
    "forward_ratio",

    "face_visible_ratio",

    "head_movement",

    "mean_confidence"
]


TARGET_COLUMN = "label"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("======================================")
    print(" CEMS ENGAGEMENT MODEL TRAINING")
    print("======================================")
    print()

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not os.path.exists(DATA_PATH):

        print(
            f"ERROR: Dataset not found: {DATA_PATH}"
        )

        sys.exit(1)

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset loaded: {DATA_PATH}"
    )

    print(
        f"Total samples: {len(data)}"
    )

    print()

    # --------------------------------------------------------
    # VALIDATE COLUMNS
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
            "ERROR: Missing columns:"
        )

        for column in missing_columns:
            print(
                f" - {column}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    data = data.dropna(
        subset=required_columns
    )

    print(
        f"Valid samples: {len(data)}"
    )

    print()

    # --------------------------------------------------------
    # CLASS COUNTS
    # --------------------------------------------------------

    print(
        "Samples per engagement class:"
    )

    class_counts = (
        data[
            TARGET_COLUMN
        ]
        .value_counts()
    )

    print(
        class_counts
    )

    print()

    if len(class_counts) < 2:

        print(
            "ERROR: Need at least two classes "
            "to train the model."
        )

        sys.exit(1)

    # Prefer all three classes.
    expected_labels = {
        "Engaged",
        "Neutral",
        "Low Engagement"
    }

    existing_labels = set(
        data[
            TARGET_COLUMN
        ].unique()
    )

    missing_labels = (
        expected_labels
        - existing_labels
    )

    if missing_labels:

        print(
            "WARNING: Some expected classes "
            "are missing:"
        )

        for label in missing_labels:
            print(
                f" - {label}"
            )

        print()

    # --------------------------------------------------------
    # WARN ABOUT SMALL DATASET
    # --------------------------------------------------------

    for (
        label,
        count
    ) in class_counts.items():

        if count < 20:

            print(
                f"WARNING: '{label}' only has "
                f"{count} samples."
            )

            print(
                "Collect more examples before "
                "treating evaluation results "
                "as reliable."
            )

            print()

    # ========================================================
    # INPUT / TARGET
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
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print()

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = (
        RandomForestClassifier(

            n_estimators=400,

            max_depth=None,

            min_samples_split=4,

            min_samples_leaf=2,

            class_weight="balanced",

            random_state=
            RANDOM_STATE,

            n_jobs=-1
        )
    )

    print(
        "Training Random Forest..."
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
    # EVALUATION
    # ========================================================

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "======================================"
    )

    print(
        " PRELIMINARY TEST RESULTS"
    )

    print(
        "======================================"
    )

    print()

    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print()

    print(
        "Classification report:"
    )

    print()

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    labels = list(
        model.classes_
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print(
        "Confusion Matrix"
    )

    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    print(
        matrix_df
    )

    print()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame(
        {
            "feature":
                FEATURE_COLUMNS,

            "importance":
                model.feature_importances_
        }
    )

    importance_df = (
        importance_df.sort_values(
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
        importance_df.to_string(
            index=False
        )
    )

    print()

    # ========================================================
    # SAVE MODEL BUNDLE
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

        "version":
            1
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
        f"Saved to: {MODEL_PATH}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This accuracy is PRELIMINARY."
    )

    print(
        "The current dataset contains overlapping "
        "temporal windows from the same recording."
    )

    print(
        "For proper final evaluation, collect "
        "separate recording sessions and test "
        "on sessions that were not used for training."
    )

    print()


if __name__ == "__main__":
    main()