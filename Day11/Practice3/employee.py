
"""
Title: Employee Productivity & Analytics Lifecycle
Description: Complete five-phase employee analytics pipeline.
"""

# =========================================================
# STEP 0: IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# =========================================================
# PHASE 1: DISCOVERY AND DATA SETUP
# =========================================================

def phase_1_discovery_and_setup():

    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)

    # Load dataset
    df = pd.read_csv(r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\Practice3\employee_productivity.csv")

    # Inspect first five records
    print("\nFirst 5 Records:")
    print(df.head())

    # Dataset dimensions
    print("\nDataset Shape:", df.shape)

    # Column names and data types
    print("\nDataset Information:")
    df.info()

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Duplicate records
    print("\nDuplicate Rows:", df.duplicated().sum())

    return df


# =========================================================
# PHASE 2: DESCRIPTIVE STATISTICS
# =========================================================

def phase_2_descriptive_statistics(df):

    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)

    column = "Weekly_Tasks_Completed"

    # Central tendency and dispersion
    mean_value = df[column].mean()
    median_value = df[column].median()
    std_value = df[column].std()

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    print(f"Mean: {mean_value:.2f}")
    print(f"Median: {median_value:.2f}")
    print(f"Standard Deviation: {std_value:.2f}")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")

    # IQR outlier detection
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    print(f"\nLower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print("\nPotential Outliers:")
    if outliers.empty:
        print("No outliers detected.")
    else:
        print(outliers[["Employee_ID", column]])

    return df


# =========================================================
# PHASE 3: BIVARIATE ASSOCIATION AND CORRELATION
# =========================================================

def phase_3_bivariate_association(df):

    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)

    selected_columns = [
        "Years_Experience",
        "Training_Hours",
        "Weekly_Tasks_Completed"
    ]

    # Select relevant numeric variables
    numeric_df = df[selected_columns]

    # Pearson correlation matrix
    correlation_matrix = numeric_df.corr(method="pearson")

    print("\nPearson Correlation Matrix:")
    print(correlation_matrix.round(3))

    # Training vs productivity correlation
    correlation = correlation_matrix.loc[
        "Training_Hours",
        "Weekly_Tasks_Completed"
    ]

    print(
        "\nCorrelation between Training Hours "
        f"and Weekly Tasks: {correlation:.3f}"
    )

    if correlation > 0:
        print("Interpretation: Positive relationship.")
    elif correlation < 0:
        print("Interpretation: Negative relationship.")
    else:
        print("Interpretation: No linear relationship.")

    return df


# =========================================================
# PHASE 4: PREDICTIVE MODELING
# =========================================================

def phase_4_predictive_modeling(df):

    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)

    # Independent variable (feature)
    X = df[["Training_Hours"]]

    # Dependent variable (target)
    y = df["Weekly_Tasks_Completed"]

    # Initialize model
    model = LinearRegression()

    # Train model
    model.fit(X, y)

    # Extract model parameters
    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")

    print(
        "\nRegression Equation:"
        f"\nWeekly Tasks = {intercept:.2f} "
        f"+ ({slope:.2f} * Training Hours)"
    )

    # Example prediction
    example = pd.DataFrame({"Training_Hours": [20]})
    prediction = model.predict(example)[0]

    print(
        f"\nPredicted weekly tasks for 20 training hours: "
        f"{prediction:.2f}"
    )

    return model, X, y


# =========================================================
# PHASE 5: EVALUATION AND HR RECOMMENDATIONS
# =========================================================

def phase_5_evaluation_and_reporting(model, X, y, df):

    print("\n[PHASE 5] Evaluation & HR Recommendations")
    print("-" * 65)

    # Predict values
    y_pred = model.predict(X)

    # Evaluation metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"R-Squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f} tasks")

    # Business interpretation
    print("\nHR Recommendations:")

    correlation = df[
        ["Training_Hours", "Weekly_Tasks_Completed"]
    ].corr().iloc[0, 1]

    if correlation >= 0.5:
        print(
            "1. Training hours show a positive association "
            "with weekly task completion."
        )
        print(
            "2. Consider targeted training programs and "
            "measure productivity before and after training."
        )
        print(
            "3. Track employee experience, task complexity, "
            "and work quality alongside task counts."
        )
    else:
        print(
            "1. Training hours have a weak or negative "
            "linear association with task completion."
        )
        print(
            "2. Investigate other productivity factors "
            "before increasing training investments."
        )

    print(
        "\nImportant: Correlation does not establish that "
        "training causes higher productivity."
    )


# =========================================================
# MAIN ORCHESTRATOR
# =========================================================

def run_analytics_lifecycle():

    print("=" * 65)
    print("STARTING EMPLOYEE ANALYTICS LIFECYCLE")
    print("=" * 65)

    # Execute all five phases
    df = phase_1_discovery_and_setup()

    df = phase_2_descriptive_statistics(df)

    df = phase_3_bivariate_association(df)

    model, X, y = phase_4_predictive_modeling(df)

    phase_5_evaluation_and_reporting(model, X, y, df)

    print("\n" + "=" * 65)
    print("EMPLOYEE ANALYTICS LIFECYCLE COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
