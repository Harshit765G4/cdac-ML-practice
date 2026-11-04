
"""
Title: Bank Loan Risk & Credit Default Prediction

Description:
End-to-end supervised classification pipeline using
Logistic Regression.

Phases:
1. Discovery and target identification
2. Feature engineering and categorical encoding
3. Train-test split and feature scaling
4. Model training and evaluation
5. Model persistence and future inference
"""

# ==========================================
# STEP 0: IMPORT LIBRARIES
# ==========================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\ML_Practice\Bank_Loan_Risk\loan_risk_data.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ==========================================
# PHASE 1: DISCOVERY & TARGET IDENTIFICATION
# ==========================================

def phase_1_discovery_and_target_identification():

    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Inspect dataset
    print("\nDataset Preview:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Identify target and predictors
    X = df.drop(columns=["applicant_id", "default_flag"])

    y = df["default_flag"]

    # Target distribution
    print("\nTarget Class Distribution:")
    print(y.value_counts())

    print("\nTarget Class Distribution (%):")
    print(y.value_counts(normalize=True) * 100)

    print("\nPredictor Features:")
    print(X.columns.tolist())

    print("\nTarget Variable:")
    print(y.name)

    return X, y


# ==========================================
# PHASE 2: FEATURE ENGINEERING & ENCODING
# ==========================================

def phase_2_feature_engineering_and_encoding(X):

    print("\n[PHASE 2] Feature Engineering & Encoding")
    print("-" * 65)

    # Create engineered feature
    X = X.copy()

    X["loan_to_income_ratio"] = (
        X["loan_amount_k"] / X["annual_income_k"]
    )

    # One-hot encoding
    X_encoded = pd.get_dummies(
        X,
        columns=["housing_status"],
        dtype=int
    )

    print("\nTransformed Dataset:")
    print(X_encoded.head())

    print("\nTransformed Shape:")
    print(X_encoded.shape)

    print("\nEncoded Feature Names:")
    print(X_encoded.columns.tolist())

    return X_encoded


# ==========================================
# PHASE 3: TRAIN-TEST SPLIT & SCALING
# ==========================================

def phase_3_scaling_and_train_test_split(X, y):

    print("\n[PHASE 3] Splitting & Feature Scaling")
    print("-" * 65)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts())

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform testing data using the same scaler
    X_test_scaled = scaler.transform(X_test)

    print("\nScaling Completed Successfully!")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# ==========================================
# PHASE 4: MODEL TRAINING & EVALUATION
# ==========================================

def phase_4_model_training_and_testing(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
):

    print("\n[PHASE 4] Model Training & Evaluation")
    print("-" * 65)

    # Initialize Logistic Regression
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    print("\nModel Training Completed!")

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("Accuracy Percentage:", accuracy * 100, "%")

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Non-Default", "Default"],
            zero_division=0
        )
    )

    return model


# ==========================================
# PHASE 5: MODEL PERSISTENCE & INFERENCE
# ==========================================

def phase_5_joblib_model_persistence(
    model,
    scaler,
    feature_names
):

    print("\n[PHASE 5] Model Persistence & Inference")
    print("-" * 65)

    # Artifact paths
    model_path = os.path.join(
        ARTIFACT_DIR,
        "loan_model.joblib"
    )

    scaler_path = os.path.join(
        ARTIFACT_DIR,
        "scaler.joblib"
    )

    features_path = os.path.join(
        ARTIFACT_DIR,
        "feature_names.joblib"
    )

    # Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)

    print("\nModel Saved:", model_path)
    print("Scaler Saved:", scaler_path)
    print("Features Saved:", features_path)

    # Load artifacts
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    loaded_features = joblib.load(features_path)

    print("\nAll Artifacts Loaded Successfully!")

    # New applicant record
    new_applicant = pd.DataFrame([{
        "age": 36,
        "annual_income_k": 60,
        "credit_score": 690,
        "employment_years": 6,
        "loan_amount_k": 25,
        "housing_status": "Rent"
    }])

    # Feature engineering
    new_applicant["loan_to_income_ratio"] = (
        new_applicant["loan_amount_k"] /
        new_applicant["annual_income_k"]
    )

    # One-hot encoding
    new_applicant = pd.get_dummies(
        new_applicant,
        columns=["housing_status"],
        dtype=int
    )

    # Match training feature columns and their order
    new_applicant = new_applicant.reindex(
        columns=loaded_features,
        fill_value=0
    )

    # Scale using the previously fitted scaler
    applicant_scaled = loaded_scaler.transform(
        new_applicant
    )

    # Predict class
    prediction = loaded_model.predict(applicant_scaled)

    # Predict probability
    probability = loaded_model.predict_proba(
        applicant_scaled
    )

    default_probability = probability[0][1]

    print("\nNEW APPLICANT PREDICTION")
    print("-" * 40)

    if prediction[0] == 1:
        print("Prediction: DEFAULT RISK")
    else:
        print("Prediction: NON-DEFAULT")

    print(
        f"Default Probability: "
        f"{default_probability * 100:.2f}%"
    )

    print(
        f"Non-Default Probability: "
        f"{probability[0][0] * 100:.2f}%"
    )

    return loaded_model


# ==========================================
# FACADE FUNCTION / ORCHESTRATOR
# ==========================================

def run_ml_pipeline():

    print("=" * 65)
    print("BANK LOAN RISK & CREDIT DEFAULT PREDICTION")
    print("=" * 65)

    # Phase 1
    X, y = phase_1_discovery_and_target_identification()

    # Phase 2
    X_encoded = phase_2_feature_engineering_and_encoding(X)

    # Phase 3
    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    ) = phase_3_scaling_and_train_test_split(
        X_encoded,
        y
    )

    # Phase 4
    model = phase_4_model_training_and_testing(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    # Phase 5
    phase_5_joblib_model_persistence(
        model,
        scaler,
        X_encoded.columns.tolist()
    )

    print("\n" + "=" * 65)
    print("LOAN RISK PIPELINE EXECUTION COMPLETED")
    print("=" * 65)


# ==========================================
# MAIN GUARD
# ==========================================

if __name__ == "__main__":
    run_ml_pipeline()
