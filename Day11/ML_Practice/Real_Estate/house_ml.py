"""
Title: End-to-End House Price Prediction
Algorithm: Linear Regression
Problem Type: Supervised Machine Learning - Regression

Pipeline:
1. Data Discovery
2. Feature Engineering and Encoding
3. Train-Test Split and Scaling
4. Model Training and Evaluation
5. Model Persistence and New Predictions
"""

# ==========================================
# STEP 0: IMPORT LIBRARIES
# ==========================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\ML_Practice\Real_Estate\housing_ml_data.csv"
ARTIFACT_DIR = "artifacts"


# ==========================================
# PHASE 1: DISCOVERY AND TARGET IDENTIFICATION
# ==========================================

def phase_1_discovery_and_target_identification():

    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Inspect dataset
    print("\nDataset Shape:", df.shape)

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Drop non-predictive identifier
    df = df.drop(columns=["house_id"])

    # Separate predictors and target
    X = df.drop(columns=["price_usd"])
    y = df["price_usd"]

    print("\nFeature Matrix Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nPredictor Columns:")
    print(X.columns.tolist())

    print("\nTarget Column: price_usd")

    return X, y


# ==========================================
# PHASE 2: FEATURE ENGINEERING AND ENCODING
# ==========================================

def phase_2_feature_engineering_and_encoding(X):

    print("\n[PHASE 2] Feature Engineering & Encoding")
    print("-" * 65)

    X = X.copy()

    # Feature engineering
    X["sqft_per_room"] = (
        X["square_feet"] / X["num_bedrooms"]
    )

    print("\nEngineered Feature: sqft_per_room")

    # One-hot encoding categorical variables
    X_encoded = pd.get_dummies(
        X,
        columns=["location_type", "has_garage"],
        drop_first=False,
        dtype=int
    )

    print("\nEncoded Feature Matrix:")
    print(X_encoded.head())

    print("\nEncoded Shape:", X_encoded.shape)

    print("\nEncoded Columns:")
    print(X_encoded.columns.tolist())

    return X_encoded


# ==========================================
# PHASE 3: TRAIN-TEST SPLIT AND SCALING
# ==========================================

def phase_3_scaling_and_train_test_split(X, y):

    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)

    print("Training Target Shape:", y_train.shape)
    print("Testing Target Shape:", y_test.shape)

    # Initialize scaler
    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature Scaling Completed.")

    print(
        "Scaled Training Mean (approximately zero):",
        np.round(X_train_scaled.mean(axis=0), 3)
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# ==========================================
# PHASE 4: MODEL TRAINING AND EVALUATION
# ==========================================

def phase_4_model_training_and_testing(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
):

    print("\n[PHASE 4] Model Training & Evaluation")
    print("-" * 65)

    # Initialize Linear Regression
    model = LinearRegression()

    # Train model
    model.fit(X_train_scaled, y_train)

    print("\nLinear Regression Model Trained Successfully.")

    # Predict on unseen test data
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Metrics:")
    print("-" * 40)

    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-Squared (R2): {r2:.4f}")

    # Compare actual and predicted values
    comparison = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": np.round(y_pred, 2),
        "Absolute Error": np.round(
            np.abs(y_test.values - y_pred), 2
        )
    })

    print("\nActual vs Predicted Prices:")
    print(comparison.to_string(index=False))

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return model, metrics


# ==========================================
# PHASE 5: MODEL PERSISTENCE AND INFERENCE
# ==========================================

def phase_5_joblib_model_persistence(
    model,
    scaler,
    feature_names
):

    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)

    # Create artifacts directory
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Define artifact paths
    model_path = os.path.join(
        ARTIFACT_DIR,
        "linear_regression_model.joblib"
    )

    scaler_path = os.path.join(
        ARTIFACT_DIR,
        "standard_scaler.joblib"
    )

    features_path = os.path.join(
        ARTIFACT_DIR,
        "feature_names.joblib"
    )

    # Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)

    print("\nModel, scaler and feature names saved.")

    # Reload artifacts
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    loaded_features = joblib.load(features_path)

    print("\nArtifacts Reloaded Successfully.")

    # New property record
    new_house = pd.DataFrame([{
        "location_type": "Urban",
        "square_feet": 2300,
        "num_bedrooms": 4,
        "building_age_years": 6,
        "has_garage": "Yes"
    }])

    # Apply the same feature engineering
    new_house["sqft_per_room"] = (
        new_house["square_feet"] /
        new_house["num_bedrooms"]
    )

    # Apply the same one-hot encoding
    new_house_encoded = pd.get_dummies(
        new_house,
        columns=["location_type", "has_garage"],
        drop_first=False,
        dtype=int
    )

    # Align columns with training data
    new_house_encoded = new_house_encoded.reindex(
        columns=loaded_features,
        fill_value=0
    )

    # Scale using the saved scaler
    new_house_scaled = loaded_scaler.transform(
        new_house_encoded
    )

    # Predict house price
    predicted_price = loaded_model.predict(
        new_house_scaled
    )[0]

    print("\nNew Property Details:")
    print(new_house.to_string(index=False))

    print("\nPredicted House Price:")
    print(f"${predicted_price:,.2f}")

    return predicted_price


# ==========================================
# FACADE FUNCTION: PIPELINE ORCHESTRATOR
# ==========================================

def run_ml_pipeline():

    print("=" * 65)
    print("STARTING HOUSE PRICE PREDICTION PIPELINE")
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
    model, metrics = phase_4_model_training_and_testing(
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
    print("HOUSE PRICE PREDICTION PIPELINE COMPLETED")
    print("=" * 65)

    return model, metrics


# ==========================================
# MAIN GUARD
# ==========================================

if __name__ == "__main__":
    run_ml_pipeline()