
"""
Title: Unsupervised Cricket Player Role Segmentation

Description:
End-to-end unsupervised machine learning pipeline using
K-Means clustering to discover cricket player archetypes.

Phases:
1. Discovery and feature selection
2. Feature scaling
3. K-Means model induction and labeling
4. Cluster evaluation and profiling
5. Model persistence and future inference
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

DATA_PATH = r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\ML_Practice\Cricket_Player_Statistics\cricket_player_stats.csv"

ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Features used for clustering
FEATURE_NAMES = [
    "runs_scored",
    "batting_average",
    "wickets_taken",
    "bowling_economy"
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
    print("\nDataset Preview:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Retain original dataset for profiling
    df_original = df.copy()

    # Select numerical performance features
    X = df[FEATURE_NAMES].copy()

    print("\nSelected Features:")
    print(X.head())

    print("\nFeature Matrix Shape:")
    print(X.shape)

    print("\nFeature Names:")
    print(X.columns.tolist())

    # Check zero bowling economy values
    print("\nPlayers with Zero Bowling Economy:")
    print(
        df.loc[
            df["bowling_economy"] == 0,
            "player_name"
        ].tolist()
    )

    return X, df_original


# ==========================================
# PHASE 2: UNSUPERVISED FEATURE SCALING
# ==========================================

def phase_2_feature_scaling(X):

    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)

    # Initialize scaler
    scaler = StandardScaler()

    # Fit and transform features
    X_scaled = scaler.fit_transform(X)

    print("\nScaled Feature Matrix:")
    print(X_scaled[:5])

    print("\nScaled Matrix Shape:")
    print(X_scaled.shape)

    print("\nFeature Means After Scaling:")
    print(X_scaled.mean(axis=0))

    print("\nFeature Standard Deviations:")
    print(X_scaled.std(axis=0))

    return X_scaled, scaler


# ==========================================
# PHASE 3: K-MEANS MODEL INDUCTION
# ==========================================

def phase_3_kmeans_model_induction(
    X_scaled,
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

    # Train model and assign cluster labels
    labels = model.fit_predict(X_scaled)

    # Add cluster labels to original dataset
    df_clustered = df_original.copy()

    df_clustered["cluster_id"] = labels

    print("\nCluster Assignments:")
    print(
        df_clustered[
            ["player_name", "cluster_id"]
        ]
    )

    print("\nCluster Distribution:")
    print(
        df_clustered["cluster_id"].value_counts().sort_index()
    )

    print("\nCluster Centroids (Scaled Space):")
    print(model.cluster_centers_)

    return model, labels, df_clustered


# ==========================================
# PHASE 4: CLUSTER EVALUATION & PROFILING
# ==========================================

def phase_4_cluster_evaluation_and_profiling(
    X_scaled,
    labels,
    df_clustered
):

    print("\n[PHASE 4] Cluster Evaluation & Role Profiling")
    print("-" * 65)

    # Calculate Silhouette Score
    score = silhouette_score(
        X_scaled,
        labels
    )

    print("\nSilhouette Score:")
    print(score)

    # Calculate average statistics per cluster
    cluster_profile = (
        df_clustered
        .groupby("cluster_id")[FEATURE_NAMES]
        .mean()
        .round(2)
    )

    print("\nCluster Performance Profiles:")
    print(cluster_profile)

    # Display players grouped by cluster
    print("\nPlayer Segmentation:")

    for cluster_id in sorted(df_clustered["cluster_id"].unique()):

        players = df_clustered.loc[
            df_clustered["cluster_id"] == cluster_id,
            "player_name"
        ].tolist()

        print(f"\nCluster {cluster_id}:")
        print(players)

    # Interpret clusters using their average statistics
    print("\nCricket Role Interpretation:")

    for cluster_id, profile in cluster_profile.iterrows():

        runs = profile["runs_scored"]
        average = profile["batting_average"]
        wickets = profile["wickets_taken"]

        if runs >= 6000 and average >= 35 and wickets < 50:

            role = "Specialist Batsmen"

        elif wickets >= 100 and average < 30:

            role = "Specialist Bowlers"

        elif runs >= 2000 and wickets >= 100:

            role = "All-Rounders"

        else:

            role = "Mixed / Other Player Profile"

        print(f"Cluster {cluster_id}: {role}")

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

    # Artifact paths
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

    print("\nArtifacts Saved Successfully!")

    # Load saved artifacts
    loaded_model = joblib.load(model_path)

    loaded_scaler = joblib.load(scaler_path)

    loaded_features = joblib.load(features_path)

    print("\nArtifacts Loaded Successfully!")

    # Create a new player profile
    new_player = pd.DataFrame([{

        "runs_scored": 4500,

        "batting_average": 34.0,

        "wickets_taken": 125,

        "bowling_economy": 5.3

    }])

    # Ensure correct feature order
    new_player = new_player.reindex(
        columns=loaded_features
    )

    # Scale new player using fitted scaler
    new_player_scaled = loaded_scaler.transform(
        new_player
    )

    # Predict cluster
    predicted_cluster = loaded_model.predict(
        new_player_scaled
    )

    print("\nNEW PLAYER SEGMENTATION")
    print("-" * 40)

    print("Player Statistics:")
    print(new_player)

    print(
        "\nAssigned Cluster:",
        predicted_cluster[0]
    )

    print(
        "\nNote: Interpret this cluster using the "
        "training cluster profile."
    )


# ==========================================
# FACADE FUNCTION / ORCHESTRATOR
# ==========================================

def run_cricket_clustering_pipeline():

    print("=" * 65)

    print(
        "STARTING CRICKET PLAYER ROLE "
        "SEGMENTATION LIFECYCLE (K-MEANS)"
    )

    print("=" * 65)

    # Phase 1
    X, df_original = phase_1_discovery_and_setup()

    # Phase 2
    X_scaled, scaler = phase_2_feature_scaling(X)

    # Phase 3
    model, labels, df_clustered = (
        phase_3_kmeans_model_induction(
            X_scaled,
            df_original
        )
    )

    # Phase 4
    cluster_profile = (
        phase_4_cluster_evaluation_and_profiling(
            X_scaled,
            labels,
            df_clustered
        )
    )

    # Phase 5
    phase_5_joblib_model_persistence(
        model,
        scaler,
        list(X.columns)
    )

    print("\n" + "=" * 65)

    print("CRICKET CLUSTERING PIPELINE COMPLETED")

    print("=" * 65)


# ==========================================
# MAIN GUARD
# ==========================================

if __name__ == "__main__":

    run_cricket_clustering_pipeline()
