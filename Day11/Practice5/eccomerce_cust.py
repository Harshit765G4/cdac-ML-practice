
"""
Title: E-Commerce Customer Analytics Lifecycle
Description: Complete five-phase data analytics pipeline.
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
    df = pd.read_csv(r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\Practice5\ecommerce_customer_data.csv")

    # Display first five records
    print("\nFirst 5 Customer Records:")
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

    # Statistical overview
    print("\nDataset Summary:")
    print(df.describe())

    return df


# =========================================================
# PHASE 2: DESCRIPTIVE STATISTICS
# =========================================================

def phase_2_descriptive_statistics(df):

    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)

    column = "Annual_Spend_USD"

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

    print(f"Mean: ${mean_value:.2f}")
    print(f"Median: ${median_value:.2f}")
    print(f"Standard Deviation: ${std_value:.2f}")
    print(f"Variance: {variance_value:.2f}")
    print(f"Q1: ${q1:.2f}")
    print(f"Q3: ${q3:.2f}")
    print(f"IQR: ${iqr:.2f}")

    # IQR outlier detection
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    print(f"\nLower Bound: ${lower_bound:.2f}")
    print(f"Upper Bound: ${upper_bound:.2f}")

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print("\nPotential Outliers:")

    if outliers.empty:
        print("No outliers detected.")
    else:
        print(outliers[["Customer_ID", column]])

    return df


# =========================================================
# PHASE 3: BIVARIATE ASSOCIATION AND CORRELATION
# =========================================================

def phase_3_bivariate_association(df):

    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)

    selected_columns = [
        "Browsing_Minutes",
        "Items_In_Cart",
        "Annual_Spend_USD"
    ]

    # Select relevant numeric variables
    numeric_df = df[selected_columns]

    # Calculate Pearson correlation
    correlation_matrix = numeric_df.corr(method="pearson")

    print("\nPearson Correlation Matrix:")
    print(correlation_matrix.round(3))

    # Correlations involving annual spending
    print("\nCorrelations with Annual Spending:")

    spend_correlations = correlation_matrix[
        "Annual_Spend_USD"
    ].drop("Annual_Spend_USD")

    print(spend_correlations.round(3))

    # Interpret correlations
    for variable, value in spend_correlations.items():

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
    X = df[["Browsing_Minutes"]]

    # Dependent variable
    y = df["Annual_Spend_USD"]

    # Initialize Linear Regression
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
        f"\nAnnual Spend = {intercept:.2f} "
        f"+ ({slope:.2f} * Browsing Minutes)"
    )

    # Example prediction
    example = pd.DataFrame({
        "Browsing_Minutes": [40]
    })

    prediction = model.predict(example)[0]

    print(
        f"\nPredicted Annual Spend for 40 Browsing Minutes: "
        f"${prediction:.2f}"
    )

    return model, X, y


# =========================================================
# PHASE 5: EVALUATION AND BUSINESS RECOMMENDATIONS
# =========================================================

def phase_5_evaluation_and_reporting(model, X, y, df):

    print("\n[PHASE 5] Evaluation & Business Action")
    print("-" * 65)

    # Generate predictions
    y_pred = model.predict(X)

    # Evaluation metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"R-Squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: ${mae:.2f}")

    # Business interpretation
    correlation = df[
        ["Browsing_Minutes", "Annual_Spend_USD"]
    ].corr().iloc[0, 1]

    print("\nBusiness Recommendations:")

    if correlation >= 0.5:

        print(
            "1. Browsing time has a strong positive association "
            "with annual customer spending."
        )

        print(
            "2. Consider personalized product recommendations "
            "for customers showing high browsing engagement."
        )

        print(
            "3. Analyze cart activity and conversion rates "
            "to identify customers who browse but do not purchase."
        )

        print(
            "4. Test targeted campaigns and measure their "
            "incremental impact on actual purchases."
        )

    else:

        print(
            "1. Browsing time has a weak or negative linear "
            "association with annual customer spending."
        )

        print(
            "2. Investigate other factors such as purchase "
            "frequency, cart size, discounts, and customer retention."
        )

    print(
        "\nImportant: Correlation does not prove that "
        "browsing time causes higher spending."
    )


# =========================================================
# MAIN ORCHESTRATOR
# =========================================================

def run_analytics_lifecycle():

    print("=" * 65)
    print("STARTING E-COMMERCE ANALYTICS LIFECYCLE")
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
    print("E-COMMERCE ANALYTICS LIFECYCLE COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
