
"""
Title: Healthcare Patient Analytics Lifecycle
Description: Complete five-phase healthcare analytics pipeline.
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
    df = pd.read_csv(r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\Practice4\patient_health_data.csv")

    # Display first five records
    print("\nFirst 5 Patient Records:")
    print(df.head())

    # Dataset dimensions
    print("\nDataset Shape:", df.shape)

    # Dataset information
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

    column = "Systolic_BP"

    # Central tendency
    mean_value = df[column].mean()
    median_value = df[column].median()

    # Dispersion
    std_value = df[column].std()
    variance_value = df[column].var()

    # Quartiles and IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    print(f"Mean: {mean_value:.2f}")
    print(f"Median: {median_value:.2f}")
    print(f"Standard Deviation: {std_value:.2f}")
    print(f"Variance: {variance_value:.2f}")
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
        print(outliers[["Patient_ID", column]])

    return df


# =========================================================
# PHASE 3: BIVARIATE ASSOCIATION AND CORRELATION
# =========================================================

def phase_3_bivariate_association(df):

    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)

    selected_columns = [
        "BMI",
        "Daily_Steps",
        "Systolic_BP"
    ]

    # Select numeric columns
    numeric_df = df[selected_columns]

    # Pearson correlation matrix
    correlation_matrix = numeric_df.corr(method="pearson")

    print("\nPearson Correlation Matrix:")
    print(correlation_matrix.round(3))

    # Correlations involving systolic BP
    print("\nCorrelations with Systolic BP:")

    bp_correlations = correlation_matrix["Systolic_BP"].drop(
        "Systolic_BP"
    )

    print(bp_correlations.round(3))

    # Interpret correlations
    for variable, value in bp_correlations.items():

        if value >= 0.7:
            strength = "Strong positive"
        elif value >= 0.3:
            strength = "Moderate positive"
        elif value > 0:
            strength = "Weak positive"
        elif value <= -0.7:
            strength = "Strong negative"
        elif value <= -0.3:
            strength = "Moderate negative"
        elif value < 0:
            strength = "Weak negative"
        else:
            strength = "No linear correlation"

        print(f"{variable}: {strength} ({value:.3f})")

    return df


# =========================================================
# PHASE 4: PREDICTIVE MODELING
# =========================================================

def phase_4_predictive_modeling(df):

    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)

    # Independent variable
    X = df[["BMI"]]

    # Dependent variable
    y = df["Systolic_BP"]

    # Initialize Linear Regression
    model = LinearRegression()

    # Train model
    model.fit(X, y)

    # Extract slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")

    print(
        "\nRegression Equation:"
        f"\nSystolic BP = {intercept:.2f} "
        f"+ ({slope:.2f} * BMI)"
    )

    # Example prediction
    example = pd.DataFrame({"BMI": [25.0]})

    prediction = model.predict(example)[0]

    print(
        f"\nEstimated Systolic BP for BMI 25: "
        f"{prediction:.2f} mmHg"
    )

    return model, X, y


# =========================================================
# PHASE 5: EVALUATION AND CLINICAL RECOMMENDATIONS
# =========================================================

def phase_5_evaluation_and_reporting(model, X, y, df):

    print("\n[PHASE 5] Evaluation & Clinical Action")
    print("-" * 65)

    # Generate predictions
    y_pred = model.predict(X)

    # Evaluation metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"R-Squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f} mmHg")

    # Clinical interpretation of observed data
    correlation = df[
        ["BMI", "Systolic_BP"]
    ].corr().iloc[0, 1]

    print("\nClinical Action Items:")

    if correlation >= 0.5:

        print(
            "1. BMI has a positive association with "
            "systolic blood pressure in this dataset."
        )

        print(
            "2. Consider monitoring BMI and blood pressure "
            "together during routine health assessments."
        )

        print(
            "3. Encourage evidence-based physical activity "
            "and healthy lifestyle interventions where appropriate."
        )

    else:

        print(
            "1. BMI shows a weak or negative linear "
            "association with systolic blood pressure."
        )

        print(
            "2. Investigate other relevant clinical factors "
            "before drawing conclusions."
        )

    print(
        "\nImportant: This model is exploratory and must not "
        "be used to diagnose hypertension or guide treatment."
    )

    print(
        "Blood pressure should be measured directly and "
        "interpreted by a qualified healthcare professional."
    )


# =========================================================
# MAIN ORCHESTRATOR
# =========================================================

def run_analytics_lifecycle():

    print("=" * 65)
    print("STARTING HEALTHCARE ANALYTICS LIFECYCLE")
    print("=" * 65)

    # Phase 1
    df = phase_1_discovery_and_setup()

    # Phase 2
    df = phase_2_descriptive_statistics(df)

    # Phase 3
    df = phase_3_bivariate_association(df)

    # Phase 4
    model, X, y = phase_4_predictive_modeling(df)

    # Phase 5
    phase_5_evaluation_and_reporting(model, X, y, df)

    print("\n" + "=" * 65)
    print("HEALTHCARE ANALYTICS LIFECYCLE COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
