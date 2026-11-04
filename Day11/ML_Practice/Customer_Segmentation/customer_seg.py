
"""
Title: Retail Customer Segmentation using K-Means
Description:
End-to-end unsupervised machine learning pipeline for customer
segmentation using income and purchasing behaviour.
"""

# ==========================================
# STEP 0: IMPORT LIBRARIES
# ==========================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\ML_Practice\Customer_Segmentation\customer_segmentation_data.csv"
ARTIFACT_DIR = "artifacts"

FEATURES = [
    "annual_income_k",
    "spending_score_1_100",
    "purchase_frequency_monthly"
]


# ==========================================
# PHASE 1: DISCOVERY & FEATURE SELECTION
# ==========================================

def phase_1_discovery_and_setup():

    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Inspect dataset
    print("\nDataset Shape:", df.shape)

    print("\nDataset Information:")
    df.info()

    print("\nFirst Five Records:")
    print(df.head())

    # Check missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Drop customer identifier
    df = df.drop(columns=["customer_id"])

    # Select relevant clustering features
    X = df[FEATURES].copy()

    print("\nSelected Features:")
    print(X.head())

    print("\nFeature Matrix Shape:", X.shape)

    print("\nFeature Names:")
    print(X.columns.tolist())

    return X, df


# ==========================================
# PHASE 2: FEATURE SCALING
# ==========================================

def phase_2_feature_scaling(X):

    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit and transform features
    X_scaled = scaler.fit_transform(X)

    print("\nOriginal Feature Means:")
    print(X.mean())

    print("\nScaled Feature Means:")
    print(X_scaled.mean(axis=0))

    print("\nScaled Feature Standard Deviations:")
    print(X_scaled.std(axis=0))

    print("\nScaled Matrix Shape:", X_scaled.shape)

    return X_scaled, scaler


# ==========================================
# PHASE 3: K-MEANS MODEL INDUCTION
# ==========================================

def phase_3_kmeans_model_induction(
    X_scaled,
    X,
    df_original
):

    print("\n[PHASE 3] K-Means Model Induction & Labeling")
    print("-" * 65)

    # Initialize K-Means
    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    # Train model and generate cluster labels
    labels = model.fit_predict(X_scaled)

    # Attach cluster labels to original data
    df_clustered = df_original.copy()

    df_clustered["cluster_id"] = labels

    print("\nCluster Assignments:")
    print(
        df_clustered[
            FEATURES + ["cluster_id"]
        ]
    )

    # Display cluster distribution
    print("\nCluster Distribution:")
    print(df_clustered["cluster_id"].value_counts().sort_index())

    # Display cluster centroids in original units
    centroids = pd.DataFrame(
        scaler_inverse_placeholder(model, X_scaled, X),
        columns=FEATURES
    )

    print("\nCluster Centroids (Original Feature Scale):")
    print(centroids)

    return model, labels, df_clustered


def scaler_inverse_placeholder(model, X_scaled, X):
    """
    Convert the scaled K-Means centroids back into original units.

    The scaler's inverse_transform is applied in Phase 4 instead.
    This helper is intentionally not used for actual conversion.
    """
    return X.groupby(
        pd.Series(
            model.labels_,
            index=X.index
        )
    ).mean().reindex(range(model.n_clusters)).values


# ==========================================
# PHASE 4: CLUSTER EVALUATION & PROFILING
# ==========================================

def phase_4_cluster_evaluation_and_profiling(
    X_scaled,
    labels,
    df_clustered,
    model,
    scaler
):

    print("\n[PHASE 4] Cluster Evaluation & Profiling")
    print("-" * 65)

    # Calculate silhouette score
    score = silhouette_score(
        X_scaled,
        labels
    )

    print("\nSilhouette Score:", round(score, 4))

    # Calculate customer counts
    print("\nCustomers in Each Cluster:")
    print(
        df_clustered["cluster_id"]
        .value_counts()
        .sort_index()
    )

    # Calculate mean feature values per cluster
    cluster_profile = (
        df_clustered
        .groupby("cluster_id")[FEATURES]
        .mean()
        .round(2)
    )

    print("\nCustomer Segment Profiles:")
    print(cluster_profile)

    # Convert centroids into original feature units
    original_centroids = scaler.inverse_transform(
        model.cluster_centers_
    )

    centroid_df = pd.DataFrame(
        original_centroids,
        columns=FEATURES
    )

    print("\nCluster Centroids:")
    print(centroid_df.round(2))

    print("\nBusiness Interpretation:")

    for cluster_id, profile in cluster_profile.iterrows():

        income = profile["annual_income_k"]
        spending = profile["spending_score_1_100"]
        frequency = profile["purchase_frequency_monthly"]

        if spending >= 70 and frequency >= 6:
            segment = "High-Spending Frequent Shoppers"

        elif income >= 65 and spending < 40:
            segment = "High-Income Conservative Shoppers"

        else:
            segment = "Balanced / Moderate Shoppers"

        print(f"Cluster {cluster_id}: {segment}")

    return cluster_profile


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
    os.makedirs(
        ARTIFACT_DIR,
        exist_ok=True
    )

    # Define paths
    model_path = os.path.join(
        ARTIFACT_DIR,
        "kmeans_model.joblib"
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

    print("\nModel, scaler and feature names saved.")

    # Load saved artifacts
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    loaded_features = joblib.load(features_path)

    print("\nArtifacts Loaded Successfully.")

    # Create a new customer
    new_customer = pd.DataFrame([{
        "annual_income_k": 35,
        "spending_score_1_100": 65,
        "purchase_frequency_monthly": 6
    }])

    # Ensure correct feature order
    new_customer = new_customer.reindex(
        columns=loaded_features
    )

    # Apply the previously fitted scaler
    new_customer_scaled = loaded_scaler.transform(
        new_customer
    )

    # Predict customer cluster
    predicted_cluster = loaded_model.predict(
        new_customer_scaled
    )

    print("\nNEW CUSTOMER SEGMENT")
    print("-" * 65)

    print("Customer Profile:")
    print(new_customer)

    print(
        "\nAssigned Cluster:",
        predicted_cluster[0]
    )


# ==========================================
# FACADE FUNCTION: PIPELINE ORCHESTRATOR
# ==========================================

def run_clustering_pipeline():

    print("=" * 65)
    print("RETAIL CUSTOMER SEGMENTATION - K-MEANS")
    print("=" * 65)

    # Phase 1
    X, df_original = phase_1_discovery_and_setup()

    # Phase 2
    X_scaled, scaler = phase_2_feature_scaling(X)

    # Phase 3
    model, labels, df_clustered = (
        phase_3_kmeans_model_induction(
            X_scaled,
            X,
            df_original
        )
    )

    # Phase 4
    phase_4_cluster_evaluation_and_profiling(
        X_scaled,
        labels,
        df_clustered,
        model,
        scaler
    )

    # Phase 5
    phase_5_joblib_model_persistence(
        model,
        scaler,
        X.columns.tolist()
    )

    print("\n" + "=" * 65)
    print("CUSTOMER SEGMENTATION PIPELINE COMPLETED")
    print("=" * 65)


# ==========================================
# MAIN GUARD
# ==========================================

if __name__ == "__main__":
    run_clustering_pipeline()
