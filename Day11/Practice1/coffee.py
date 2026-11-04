"""
Title: End-to-End Data Analytics Lifecycle
Project: Coffee Shop Daily Revenue Analysis

Business Objective:
Analyze the relationship between temperature, foot traffic,
marketing expenditure and daily revenue.

Phases:
1. Discovery and Business Understanding
2. Descriptive Statistics
3. Bivariate and Multivariate Association
4. Predictive Modeling
5. Model Evaluation and Business Recommendations
"""

# ==========================================
# IMPORTS
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\Practice1\coffee_shop_data.csv"
OUTPUT_DIR = "outputs"

FEATURES = [
    "Temperature_C",
    "Foot_Traffic",
    "Marketing_Spend_USD"
]

TARGET = "Daily_Revenue_USD"


# ==========================================
# PHASE 1: DISCOVERY AND BUSINESS UNDERSTANDING
# ==========================================

def phase_1_discovery_and_setup():

    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Initial inspection
    print("\nDataset Shape:", df.shape)

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:", df.duplicated().sum())

    # Business understanding
    print("\nBUSINESS OBJECTIVE:")
    print(
        "Understand how temperature, foot traffic and marketing "
        "spend relate to daily coffee shop revenue."
    )

    print("\nTARGET VARIABLE:", TARGET)

    print("PREDICTOR VARIABLES:", FEATURES)

    print("\nBasic Data Quality Check:")

    if df.isnull().sum().sum() == 0:
        print("No missing values found.")

    if df.duplicated().sum() == 0:
        print("No duplicate records found.")

    return df


# ==========================================
# PHASE 2: DESCRIPTIVE STATISTICS
# ==========================================

def phase_2_descriptive_statistics(df):

    print("\n[PHASE 2] Descriptive Statistics & Measures")
    print("-" * 65)

    numeric_df = df.select_dtypes(include=np.number)

    # Central tendency
    print("\n1. MEAN:")
    print(numeric_df.mean().round(2))

    print("\n2. MEDIAN:")
    print(numeric_df.median().round(2))

    print("\n3. MODE:")
    for column in numeric_df.columns:
        modes = numeric_df[column].mode().tolist()
        print(f"{column}: {modes}")

    # Dispersion
    print("\n4. STANDARD DEVIATION:")
    print(numeric_df.std().round(2))

    print("\n5. VARIANCE:")
    print(numeric_df.var().round(2))

    print("\n6. RANGE:")

    for column in numeric_df.columns:
        data_range = (
            numeric_df[column].max() -
            numeric_df[column].min()
        )

        print(f"{column}: {data_range:.2f}")

    # Interquartile range
    print("\n7. INTERQUARTILE RANGE (IQR):")

    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)

    IQR = Q3 - Q1

    print(IQR.round(2))

    # Outlier detection using IQR
    print("\n8. OUTLIER DETECTION:")

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    for column in numeric_df.columns:

        outliers = numeric_df[
            (numeric_df[column] < lower_bound[column]) |
            (numeric_df[column] > upper_bound[column])
        ][column]

        print(f"\n{column}:")
        print("Outlier Count:", len(outliers))
        print("Outlier Values:", outliers.tolist())

    # Distribution visualization
    plt.figure(figsize=(12, 7))

    numeric_df.hist(
        figsize=(12, 8),
        bins=10,
        edgecolor="black"
    )

    plt.suptitle("Distribution of Coffee Shop Variables")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "variable_distributions.png"),
        dpi=150
    )

    plt.show()

    # Boxplots
    plt.figure(figsize=(12, 6))

    sns.boxplot(data=numeric_df)

    plt.title("Boxplot - Outlier Inspection")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "outlier_boxplots.png"),
        dpi=150
    )

    plt.show()

    # Important: We report potential outliers but do not
    # automatically delete records.

    print("\nOutlier inspection completed.")
    print("No records were automatically removed.")


# ==========================================
# PHASE 3: BIVARIATE AND MULTIVARIATE ASSOCIATION
# ==========================================

def phase_3_bivariate_association(df):

    print("\n[PHASE 3] Bivariate Association & Visualization")
    print("-" * 65)

    # Exclude Day because it is an identifier
    numeric_df = df[FEATURES + [TARGET]]

    # Covariance matrix
    print("\nCOVARIANCE MATRIX:")

    covariance_matrix = numeric_df.cov()

    print(covariance_matrix.round(2))

    # Pearson correlation
    print("\nPEARSON CORRELATION MATRIX:")

    correlation_matrix = numeric_df.corr(method="pearson")

    print(correlation_matrix.round(4))

    # Correlation heatmap
    plt.figure(figsize=(9, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap - Coffee Shop Analytics")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
        dpi=150
    )

    plt.show()

    # Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for i, feature in enumerate(FEATURES):

        sns.regplot(
            data=df,
            x=feature,
            y=TARGET,
            ax=axes[i],
            scatter_kws={"alpha": 0.7}
        )

        axes[i].set_title(f"{feature} vs Revenue")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "revenue_scatterplots.png"),
        dpi=150
    )

    plt.show()

    # Rank relationships with revenue
    revenue_correlations = (
        correlation_matrix[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
    )

    print("\nCORRELATION WITH DAILY REVENUE:")
    print(revenue_correlations.round(4))

    return correlation_matrix


# ==========================================
# PHASE 4: PREDICTIVE MODELING
# ==========================================

def phase_4_predictive_modeling(df):

    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)

    # Independent variables
    X = df[FEATURES]

    # Dependent variable
    y = df[TARGET]

    print("\nFeature Matrix Shape:", X.shape)
    print("Target Shape:", y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining Records:", len(X_train))
    print("Testing Records:", len(X_test))

    # Initialize model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    print("\nMultiple Linear Regression Trained Successfully.")

    # Coefficients
    print("\nMODEL INTERCEPT:")
    print(round(model.intercept_, 4))

    print("\nMODEL COEFFICIENTS:")

    coefficient_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_
    })

    print(coefficient_df.to_string(index=False))

    # Predict test records
    y_pred = model.predict(X_test)

    # Actual vs predicted
    comparison = pd.DataFrame({
        "Actual_Revenue": y_test.values,
        "Predicted_Revenue": np.round(y_pred, 2),
        "Absolute_Error": np.round(
            np.abs(y_test.values - y_pred),
            2
        )
    })

    print("\nACTUAL VS PREDICTED REVENUE:")
    print(comparison.to_string(index=False))

    # Prediction visualization
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.8
    )

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Revenue ($)")
    plt.ylabel("Predicted Revenue ($)")
    plt.title("Actual vs Predicted Daily Revenue")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "revenue_prediction.png"),
        dpi=150
    )

    plt.show()

    return model, X_train, X_test, y_train, y_test, y_pred


# ==========================================
# PHASE 5: EVALUATION AND BUSINESS REPORTING
# ==========================================

def phase_5_evaluation_and_reporting(
    model,
    y_test,
    y_pred,
    correlation_matrix
):

    print("\n[PHASE 5] Model Evaluation & Business Action")
    print("-" * 65)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    print("\nMODEL PERFORMANCE:")

    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    # Strongest observed association with revenue
    strongest_factor = (
        correlation_matrix[TARGET]
        .drop(TARGET)
        .abs()
        .idxmax()
    )

    strongest_correlation = correlation_matrix.loc[
        strongest_factor,
        TARGET
    ]

    print("\nBUSINESS INSIGHTS:")

    print(
        f"Strongest linear association with revenue: "
        f"{strongest_factor}"
    )

    print(
        f"Pearson correlation: {strongest_correlation:.4f}"
    )

    print("\nBUSINESS RECOMMENDATIONS:")

    print("1. Monitor daily foot traffic and revenue together.")

    print("2. Compare marketing expenditure against revenue changes.")

    print("3. Track temperature and seasonal demand patterns.")

    print("4. Evaluate marketing campaigns using incremental revenue "
          "and profitability, not correlation alone.")

    print("5. Collect more historical data before deploying "
          "the model for business decisions.")

    print("\nGENERALIZATION AND OVERFITTING:")

    print(
        "The dataset contains only 25 records and the test set "
        "contains only 5 records."
    )

    print(
        "Therefore, evaluation metrics may change substantially "
        "on a larger dataset."
    )

    print(
        "Correlation does not establish causation. "
        "The model cannot prove that marketing spend causes "
        "higher revenue."
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ==========================================
# FACADE FUNCTION: COMPLETE LIFECYCLE
# ==========================================

def run_analytics_lifecycle():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("STARTING COMPLETE DATA ANALYTICS LIFECYCLE")
    print("=" * 65)

    # Phase 1
    df = phase_1_discovery_and_setup()

    print("-" * 65)

    # Phase 2
    phase_2_descriptive_statistics(df)

    print("-" * 65)

    # Phase 3
    correlation_matrix = phase_3_bivariate_association(df)

    print("-" * 65)

    # Phase 4
    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred
    ) = phase_4_predictive_modeling(df)

    print("-" * 65)

    # Phase 5
    metrics = phase_5_evaluation_and_reporting(
        model,
        y_test,
        y_pred,
        correlation_matrix
    )

    print("\nFINAL METRICS SUMMARY:")
    print(metrics)

    print("=" * 65)
    print("ANALYTICS LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 65)


# ==========================================
# MAIN GUARD
# ==========================================

if __name__ == "__main__":
    run_analytics_lifecycle()