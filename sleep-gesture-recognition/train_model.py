#TRAINING MODEL SCRIPT
# Takes the CSV dataset, then
# an SVM is used for classifying the poses, GridSearchCV is used to find the best parameters
# The model is evaluated for accuracy and other parameters + confusion matrix
# Finally, the model is saved, so it can be used in the live inference

# IMPORTS
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# settings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sleep_gesture_data.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)

# define labels
LABELS = [
    "neutral",
    "sleep_left",
    "sleep_right",
    "hand_near_face_no_sleep",
    "head_lean_only",
]

RANDOM_STATE = 42 # for recreation purposes
TEST_SIZE = 0.2 # test split

# load dataset and do basic validation to prevent crashes
def load_dataset(csv_path: Path) -> pd.DataFrame:

    # check path
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # check structure
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    # if CSV is empty
    if df.empty:
        raise ValueError("CSV is empty.")

    return df

# split the dataset into X and y
def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    X = df.drop(columns=["label"]).copy() # features
    y = df["label"].copy() # labels

    # convert all columns to numbers
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist() # convert anything that fails to NaN
        raise ValueError(f"Found non-numeric or missing values in columns: {bad_cols}")

    return X, y

# train model
def train_and_evaluate(df: pd.DataFrame) -> None:
    # only keep labels we expect
    df = df[df["label"].isin(LABELS)].copy()

    # handle empty dataframe
    if df.empty:
        raise ValueError("No rows found for the expected sleep gesture labels.")

    # print state
    print("\n--- Training sleep gesture model ---")
    print("Class counts:")
    print(df["label"].value_counts())

    # safety check that every label has at least 2 samples
    class_counts = df["label"].value_counts()
    too_small = class_counts[class_counts < 2]

    if not too_small.empty:
        raise ValueError(
            f"These labels have fewer than 2 samples and cannot be split safely:\n{too_small}"
        )

    # train and test splits
    X, y = split_features_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # create pipeline
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True)),
        ]
    )

    # parameters for GridSearch
    param_grid = [
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1, 10, 100],
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.01, 0.1, 1],
        },
    ]

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    # run GridSearch to find best parameters
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # evaluation metrics (accuracy and confusion matrix)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    # print evaluation of model and save model
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")
    print(f"Test accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=LABELS))

    model_path = MODEL_DIR / "sleep_gesture_svm.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved model to: {model_path}")

    # create confusion matrix and save it
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title("Sleep gesture confusion matrix")
    fig.tight_layout()

    fig_path = MODEL_DIR / "sleep_gesture_confusion_matrix.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved confusion matrix to: {fig_path}")


def main() -> None:
    df = load_dataset(DATA_PATH)

    print("Loaded dataset shape:", df.shape)
    print("\nAll labels in dataset:")
    print(df["label"].value_counts())

    train_and_evaluate(df)

    print("\nDone.")


if __name__ == "__main__":
    main()