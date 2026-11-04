"""
Title: End-to-End Data Analytics Lifecycle
Description: A modular Python script demonstrating business discovery,
             data preparation, exploration, statistics, and predictive modeling.
"""

# ==========================================
# IMPORTS (Libraries for Data, Stats, & Viz)
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# ==========================================
# SUB-STEP FUNCTIONS (The Lifecycle Phases)
# ==========================================

df = pd.read_csv("coffee_shop_data.csv")
df_numeric = df.select_dtypes(include=[np.number])


def phase_1_discovery_and_setup():
    """
    Module 1: Foundations, Business Understanding & Setup
    - Define business objective (e.g., Coffee Shop Revenue Analysis).
    - Load the 25-record real-world CSV dataset.
    - Initial inspection (head, info, missing values).

    """
    print("Executing Phase 1: Discovery & Data Setup...")
    # TODO: Load dataset using pandas
    # TODO: Perform basic inspection (shape, data types, missing values)

    # print("-" * 60)
    # print(df.info)
    # print("-" * 60)
    # print(df.isna())


def phase_2_descriptive_statistics():
    """
    Module 2: Descriptive Statistics & Data Measures
    - Calculate Central Tendency (Mean, Median, Mode).
    - Calculate Dispersion (Standard Deviation, Variance, IQR, Range).
    - Check Distribution Shape & Outliers.

    """
    print("Executing Phase 2: Descriptive Statistics & Measures...")
    # TODO: Compute summary statistics (mean, median, std, IQR)
    # TODO: Identify and handle any Outliers

    # print("Mean Daily Revenue: \n", df["Daily_Revenue_USD"].mean())
    # print("-" * 60)
    # print("Mean Marketing Revenue : \n", df["Marketing_Spend_USD"].mean())
    # print("-" * 60)
    # print("Median Daily Revenue: \n", df["Daily_Revenue_USD"].median())
    # print("-" * 60)
    # print("Median Marketing Revenue : \n", df["Marketing_Spend_USD"].median())
    # print("-" * 60)
    # print("Std Daily Revenue: \n", df["Daily_Revenue_USD"].std())
    # print("-" * 60)
    # print("Std Daily Revenue: \n", df["Marketing_Spend_USD"].std())
    # print("-" * 60)


def phase_3_bivariate_association():
    """
    Module 4: Bivariate & Multivariate Association
    - Analyze relationships between variables (e.g., Marketing Spend vs. Revenue).
    - Compute Covariance and Pearson Correlation Matrix.
    - Generate exploratory visualizations (Scatter plots, Correlation heatmap).

    """
    print("Executing Phase 3: Bivariate Association & Visualization...")
    # TODO: Compute correlation matrix
    # TODO: Plot correlation heatmap and scatter plots

    pearson_coffee_corr = df_numeric.corr(method="pearson")
    kendall_coffee_corr = df_numeric.corr(method="kendall")
    covariance = df_numeric.cov()

    print("Pearson Corr: \n", pearson_coffee_corr)
    print("-" * 60)
    print("Kendall Corr: \n", kendall_coffee_corr)
    print("-" * 60)
    print("Covariance \n", covariance)

    plt.figure(figsize=(7, 4))

    sns.scatterplot(data=df, x="Marketing_Spend_USD", y="Daily_Revenue_USD")
    plt.title("Daily Revenue VS Marketing Spend Scatterplot")
    plt.legend("Marketing_Spend_USD", "Daily_Revenue_USD")
    plt.show()

    plt.figure(figsize=(5, 3))
    sns.heatmap(
        pearson_coffee_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1.5
    )

    plt.title("Pearson heatmap")
    plt.show()


def phase_4_predictive_modeling():
    """
    Module 6 & 8: Predictive Analytics & Regression Modeling
    - Set up a simple predictive model (e.g., Simple Linear Regression).
    - Train the model on independent variables to predict revenue.
    - Establish a baseline evaluation.

    """
    print("Executing Phase 4: Predictive Modeling...")
    # TODO: Define features (X) and target (y)
    # TODO: Fit a simple linear regression model
    # TODO: Extract model coefficients and intercept
    pass


def phase_5_evaluation_and_reporting():
    """
    Module 6 & 7: Model Evaluation, Generalization & Business Decision
    - Evaluate model performance (e.g., R-squared, Mean Absolute Error).
    - Discuss generalization and overfitting risks on small data.
    - Translate technical findings into prescriptive business recommendations.
    [Allocation: 5 Marks]
    """
    print("Executing Phase 5: Evaluation & Business Action...")
    # TODO: Calculate evaluation metrics
    # TODO: Print final business recommendations based on findings
    pass


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================


def run_analytics_lifecycle():
    """
    Facade function that coordinates the entire data analytics
    lifecycle in chronological order.
    """
    print("=" * 60)
    print("STARTING COMPLETE DATA ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 60)

    phase_1_discovery_and_setup()
    print("-" * 60)

    phase_2_descriptive_statistics()
    print("-" * 60)

    phase_3_bivariate_association()
    print("-" * 60)

    phase_4_predictive_modeling()
    print("-" * 60)

    phase_5_evaluation_and_reporting()

    print("=" * 60)
    print("ANALYTICS LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_analytics_lifecycle()
