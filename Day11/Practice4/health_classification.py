
"""
Title: Healthcare Classification Lifecycle
Description: Module 7 - Supervised Classification Exercise.
"""

# =========================================================
# STEP 0: IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# PHASE 1: DISCOVERY AND DATA SETUP
# =========================================================

def phase_1_discovery_and_setup():

    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)

    # Load dataset
    df = pd.read_csv(r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\Practice4\patient_classification_data.csv")

    # Display records
    print("\nFirst 5 Records:")
    print(df.head())

    # Dataset dimensions
    print("\nDataset Shape:", df.shape)

    # Dataset information
    print("\nDataset Information:")
    df.info()

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Class distribution
    print("\nHypertension Risk Distribution:")
    print(df["High_Risk_Hypertension"].value_counts().sort_index())

    # Class distribution in percentage
    print("\nClass Distribution (%):")
    print(
        df["High_Risk_Hypertension"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # Duplicate records
    print("\nDuplicate Rows:", df.duplicated().sum())

    return df


# =========================================================
# PHASE 2: SUPERVISED SEGMENTATION
# =========================================================

def phase_2_supervised_segmentation(df):

    print("\n[PHASE 2] Supervised Segmentation")
    print("-" * 65)

    # Group patients according to their risk class
    grouped_data = df.groupby("High_Risk_Hypertension")

    # Calculate average BMI and Daily Steps
    group_summary = grouped_data[
        ["BMI", "Daily_Steps"]
    ].mean()

    print("\nAverage Feature Values by Risk Group:")
    print(group_summary.round(2))

    # Count patients in each group
    print("\nNumber of Patients in Each Group:")
    print(grouped_data.size())

    # Interpret group differences
    print("\nGroup Interpretation:")

    print(
        "Not High Risk: "
        f"Average BMI = {group_summary.loc[0, 'BMI']:.2f}, "
        f"Average Steps = {group_summary.loc[0, 'Daily_Steps']:.0f}"
    )

    print(
        "High Risk: "
        f"Average BMI = {group_summary.loc[1, 'BMI']:.2f}, "
        f"Average Steps = {group_summary.loc[1, 'Daily_Steps']:.0f}"
    )

    return df


# =========================================================
# PHASE 3: CLASSIFICATION MODELING
# =========================================================

def phase_3_classification_modeling(df):

    print("\n[PHASE 3] Classification Modeling")
    print("-" * 65)

    # Independent variables / features
    X = df[["BMI", "Daily_Steps"]]

    # Dependent variable / target
    y = df["High_Risk_Hypertension"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    # Build a pipeline:
    # 1. StandardScaler scales the features.
    # 2. LogisticRegression trains the classifier.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    print("\nLogistic Regression Model Trained Successfully.")

    return model, X_test, y_test


# =========================================================
# PHASE 4: MODEL EVALUATION AND CLINICAL ACTION
# =========================================================

def phase_4_model_evaluation(model, X, y):

    print("\n[PHASE 4] Model Evaluation & Clinical Action")
    print("-" * 65)

    # Generate predictions
    y_pred = model.predict(X)

    # Accuracy
    accuracy = accuracy_score(y, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"Model Accuracy (%): {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=["Not High Risk", "High Risk"],
            zero_division=0
        )
    )

    # Extract confusion matrix values
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix Interpretation:")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    # Clinical recommendations
    print("\nClinical Action Items:")

    print(
        "1. Patients classified as high-risk should receive "
        "appropriate clinical assessment."
    )

    print(
        "2. False negatives must be investigated because "
        "a high-risk patient may be incorrectly classified."
    )

    print(
        "3. Use direct blood pressure measurements and "
        "professional clinical assessment for decisions."
    )

    print(
        "4. Validate the model on a larger, independent "
        "patient dataset before considering real-world use."
    )

    print(
        "\nImportant: This educational classifier is not "
        "a diagnostic medical device."
    )


# =========================================================
# MAIN ORCHESTRATOR
# =========================================================

def run_analytics_lifecycle():

    print("=" * 65)
    print("STARTING HEALTHCARE CLASSIFICATION LIFECYCLE")
    print("=" * 65)

    # Phase 1
    df = phase_1_discovery_and_setup()

    # Phase 2
    df = phase_2_supervised_segmentation(df)

    # Phase 3
    model, X_test, y_test = phase_3_classification_modeling(df)

    # Phase 4
    phase_4_model_evaluation(model, X_test, y_test)

    print("\n" + "=" * 65)
    print("HEALTHCARE CLASSIFICATION LIFECYCLE COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
