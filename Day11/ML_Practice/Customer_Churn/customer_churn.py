
"""
Title: Customer Churn Prediction using Random Forest
Description:
End-to-end supervised machine learning pipeline for predicting
customer churn using Random Forest classification.
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

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\ML_Practice\Customer_Churn\customer_churn_data.csv"
ARTIFACT_DIR = "artifacts"


# ==========================================
# PHASE 1: DISCOVERY & TARGET IDENTIFICATION
# ==========================================

def phase_1_discovery_and_target_identification():

    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Basic inspection
    print("\nDataset Shape:", df.shape)

    print("\nDataset Information:")
    df.info()

    print("\nFirst Five Records:")
    print(df.head())

    # Check missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Check target distribution
    print("\nChurn Distribution:")
    print(df["churn_flag"].value_counts())

    print("\nChurn Percentage:")
    print(df["churn_flag"].value_counts(normalize=True) * 100)

    # Remove customer identifier
    df = df.drop(columns=["customer_id"])

    # Separate predictors and target
    X = df.drop(columns=["churn_flag"])
    y = df["churn_flag"]

    print("\nPredictor Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nPredictor Columns:")
    print(X.columns.tolist())

    return X, y


# ==========================================
# PHASE 2: FEATURE ENGINEERING & ENCODING
# ==========================================

def phase_2_feature_engineering_and_encoding(X):

    print("\n[PHASE 2] Feature Engineering & Encoding")
    print("-" * 65)

    # Make a copy to avoid modifying original data
    X = X.copy()

    # Create engineered feature
    X["avg_monthly_spend"] = (
        X["total_charges"] / X["tenure_months"]
    )

    print("\nEngineered Feature Created:")
    print("avg_monthly_spend")

    # Identify categorical columns
    categorical_columns = [
        "contract_type",
        "internet_service"
    ]

    # One-hot encoding
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=int
    )

    print("\nEncoded Dataset:")
    print(X_encoded.head())

    print("\nEncoded Shape:", X_encoded.shape)

    print("\nEncoded Feature Names:")
    print(X_encoded.columns.tolist())

    return X_encoded


# ==========================================
# PHASE 3: TRAIN-TEST SPLIT & SCALING
# ==========================================

def phase_3_scaling_and_train_test_split(X, y):

    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Samples:", X_train.shape[0])
    print("Testing Samples:", X_test.shape[0])

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts())

    # Initialize scaler
    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)

    print("\nScaling Completed.")

    print("Scaled Training Shape:", X_train_scaled.shape)
    print("Scaled Testing Shape:", X_test_scaled.shape)

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

    # Initialize Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    print("\nRandom Forest Training Completed.")

    # Generate predictions
    y_pred = model.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Accuracy:", accuracy)

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Not Churned", "Churned"],
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

    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)

    # Create artifact directory
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Define artifact paths
    model_path = os.path.join(
        ARTIFACT_DIR,
        "random_forest_model.joblib"
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

    print("\nModel, scaler and features saved successfully.")

    # Reload artifacts
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    loaded_features = joblib.load(features_path)

    print("\nArtifacts Loaded Successfully.")

    # Create new customer record
    new_customer = pd.DataFrame([{
        "tenure_months": 3,
        "monthly_charges": 85.0,
        "total_charges": 255.0,
        "support_tickets": 4,
        "contract_type": "Month-to-Month",
        "internet_service": "Fiber Optic"
    }])

    # Feature engineering
    new_customer["avg_monthly_spend"] = (
        new_customer["total_charges"] /
        new_customer["tenure_months"]
    )

    # One-hot encoding
    new_customer_encoded = pd.get_dummies(
        new_customer,
        columns=["contract_type", "internet_service"],
        dtype=int
    )

    # Align new record with training features
    new_customer_encoded = new_customer_encoded.reindex(
        columns=loaded_features,
        fill_value=0
    )

    # Scale new customer
    new_customer_scaled = loaded_scaler.transform(
        new_customer_encoded
    )

    # Predict churn
    prediction = loaded_model.predict(new_customer_scaled)

    # Predict probability of churn
    probabilities = loaded_model.predict_proba(
        new_customer_scaled
    )

    # Find probability corresponding to class 1
    churn_class_index = list(
        loaded_model.classes_
    ).index(1)

    churn_probability = probabilities[0][churn_class_index]

    # Display result
    print("\nNEW CUSTOMER PREDICTION")
    print("-" * 65)

    if prediction[0] == 1:
        print("Prediction: Customer Likely to Churn")
    else:
        print("Prediction: Customer Likely to Stay")

    print(
        f"Churn Risk Probability: "
        f"{churn_probability * 100:.2f}%"
    )


# ==========================================
# FACADE FUNCTION: PIPELINE ORCHESTRATOR
# ==========================================

def run_ml_pipeline():

    print("=" * 65)
    print("CUSTOMER CHURN PREDICTION - RANDOM FOREST")
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
    print("CUSTOMER CHURN PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)


# ==========================================
# MAIN GUARD
# ==========================================

if __name__ == "__main__":
    run_ml_pipeline()
