"""
Title: NBA Player Biographical & Birthday Paradox Analysis

Description:
A modular Python application for:
1. NBA player data discovery and sanitization
2. Birthday distribution analysis
3. Birthday Paradox probability
4. Player biographical lookup
5. Shared birthday cohort analysis
6. Career duration summary
"""

# ==========================================
# STEP 0: IMPORT LIBRARIES
# ==========================================

import os
import re
import pandas as pd
import numpy as np

from datetime import datetime


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\Practice2\player_data.csv"
OUTPUT_DIR = r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\Day11\Practice2"

TARGET_PLAYER = "Kareem Abdul-Jabbar"


# ==========================================
# HELPER FUNCTION: NORMALIZE PLAYER NAMES
# ==========================================

def normalize_player_name(name):

    if pd.isna(name):
        return ""

    # Remove leading/trailing whitespace
    name = str(name).strip()

    # Remove trailing asterisk used for Hall of Fame markers
    name = re.sub(r"\*+$", "", name)

    # Normalize case and internal whitespace
    name = " ".join(name.split()).casefold()

    return name


# ==========================================
# PHASE 1: DISCOVERY AND SANITIZATION
# ==========================================

def phase_1_discovery_and_sanitization():

    print("\n[PHASE 1] Data Discovery & Sanitization")
    print("-" * 65)

    # Load CSV
    df = pd.read_csv(DATA_PATH)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Convert birth_date into datetime
    df["birth_date"] = pd.to_datetime(
        df["birth_date"],
        errors="coerce"
    )

    # Convert career years into numeric values
    df["from"] = pd.to_numeric(
        df["from"],
        errors="coerce"
    )

    df["to"] = pd.to_numeric(
        df["to"],
        errors="coerce"
    )

    # Clean player names
    df["player"] = df["player"].astype("string").str.strip()

    # Create normalized player name for searching
    df["normalized_player"] = df["player"].apply(
        normalize_player_name
    )

    # Inspect dataset
    print("\nDataset Shape:", df.shape)

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print(
        "\nMissing Birth Dates:",
        df["birth_date"].isnull().sum()
    )

    print(
        "\nDuplicate Player Names:",
        df["normalized_player"].duplicated().sum()
    )

    print("\nData Sanitization Completed.")

    return df


# ==========================================
# PHASE 2: BIRTHDAY PARADOX ANALYSIS
# ==========================================

def phase_2_birthday_paradox_analysis(df):

    print("\n[PHASE 2] Birthday Paradox & Distribution")
    print("-" * 65)

    # Work only with valid birth dates
    valid_birthdays = df.dropna(
        subset=["birth_date"]
    ).copy()

    # Extract month and day
    valid_birthdays["birth_month"] = (
        valid_birthdays["birth_date"].dt.month
    )

    valid_birthdays["birth_day"] = (
        valid_birthdays["birth_date"].dt.day
    )

    # Create birthday key without birth year
    valid_birthdays["birthday_key"] = (
        valid_birthdays["birth_date"].dt.strftime("%m-%d")
    )

    # ======================================
    # 1. Birthday distribution by month
    # ======================================

    print("\n1. PLAYERS BORN IN EACH MONTH:")

    month_counts = (
        valid_birthdays["birth_month"]
        .value_counts()
        .sort_index()
    )

    month_names = {
        i: datetime(2000, i, 1).strftime("%B")
        for i in range(1, 13)
    }

    month_distribution = pd.DataFrame({
        "Month": [
            month_names[i] for i in month_counts.index
        ],
        "Player_Count": month_counts.values
    })

    print(month_distribution.to_string(index=False))

    # ======================================
    # 2. Birthday frequency by calendar day
    # ======================================

    print("\n2. SHARED BIRTHDAYS (MONTH AND DAY):")

    birthday_counts = (
        valid_birthdays["birthday_key"]
        .value_counts()
        .sort_values(ascending=False)
    )

    shared_birthdays = birthday_counts[
        birthday_counts > 1
    ]

    if len(shared_birthdays) > 0:

        for birthday, count in shared_birthdays.items():

            players = valid_birthdays.loc[
                valid_birthdays["birthday_key"] == birthday,
                "player"
            ].tolist()

            print(
                f"{birthday}: {count} players -> {players}"
            )

    else:
        print("No shared calendar birthdays found.")

    # ======================================
    # 3. Birthday Paradox Probability
    # ======================================

    n = len(valid_birthdays)

    print("\n3. BIRTHDAY PARADOX PROBABILITY:")

    print("Number of valid players:", n)

    # Standard theoretical model:
    # Assumes 365 equally likely birthdays,
    # independent observations and ignores leap day.

    probability_no_match = 1.0

    for i in range(n):

        probability_no_match *= (365 - i) / 365

    probability_match = 1 - probability_no_match

    print(
        f"Theoretical probability of at least one "
        f"shared birthday among {n} people: "
        f"{probability_match:.4%}"
    )

    # Show how probability changes with group size
    print("\n4. PROBABILITY BY GROUP SIZE:")

    probability_rows = []

    for group_size in range(2, min(n, 50) + 1):

        no_match = 1.0

        for i in range(group_size):

            no_match *= (365 - i) / 365

        match_probability = 1 - no_match

        probability_rows.append({
            "Group_Size": group_size,
            "Shared_Birthday_Probability": match_probability
        })

    probability_df = pd.DataFrame(probability_rows)

    probability_df["Shared_Birthday_Probability"] = (
        probability_df["Shared_Birthday_Probability"] * 100
    ).round(2)

    print(probability_df.to_string(index=False))

    # ======================================
    # 5. Visualizations
    # ======================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Month distribution
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(11, 5))

    sns.barplot(
        data=month_distribution,
        x="Month",
        y="Player_Count"
    )

    plt.title("NBA Player Birthdays by Month")
    plt.xlabel("Birth Month")
    plt.ylabel("Number of Players")
    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "birthdays_by_month.png"),
        dpi=150
    )

    plt.show()

    # Birthday frequency
    plt.figure(figsize=(12, 5))

    birthday_counts.sort_index().plot(kind="bar")

    plt.title("NBA Player Birthday Frequency")
    plt.xlabel("Month-Day")
    plt.ylabel("Number of Players")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "birthday_frequency.png"),
        dpi=150
    )

    plt.show()

    print("\nBirthday analysis completed.")

    return valid_birthdays


# ==========================================
# PHASE 3: PLAYER LOOKUP SYSTEM
# ==========================================

def phase_3_player_lookup_system(df, player_name):

    print(
        f"\n[PHASE 3] Player Lookup: {player_name}"
    )

    print("-" * 65)

    search_name = normalize_player_name(player_name)

    # First try exact normalized match
    exact_match = df[
        df["normalized_player"] == search_name
    ]

    if not exact_match.empty:

        result = exact_match.iloc[0]

    else:

        # Partial name search
        partial_match = df[
            df["normalized_player"].str.contains(
                re.escape(search_name),
                na=False
            )
        ]

        if partial_match.empty:

            print("Player not found.")

            return None

        if len(partial_match) > 1:

            print("\nMultiple matching players found:")

            print(
                partial_match["player"].to_string(index=False)
            )

            print(
                "Please provide a more specific player name."
            )

            return None

        result = partial_match.iloc[0]

    # Display player biography
    print("\nPLAYER BIOGRAPHICAL DETAILS")
    print("-" * 40)

    print("Player:", result["player"])
    print("Position:", result["pos"])
    print("Height:", result["ht"])
    print("Weight:", result["wt"])
    print("College:", result["colleges"])
    print("Birth Date:", result["birth_date"].strftime("%B %d, %Y"))

    print("Career From:", result["from"])
    print("Career To:", result["to"])

    return result


# ==========================================
# PHASE 4: SHARED BIRTHDAY COHORT
# ==========================================

def phase_4_shared_birthday_cohort(df, player_name):

    print(
        f"\n[PHASE 4] Shared Birthday Cohort: {player_name}"
    )

    print("-" * 65)

    # Find target player
    player = phase_3_player_lookup_system(
        df,
        player_name
    )

    if player is None:
        return None

    target_birth_date = player["birth_date"]

    if pd.isna(target_birth_date):

        print("Target player has no valid birth date.")

        return None

    # ======================================
    # Compare month and day, ignoring year
    # ======================================

    same_birthday = df[
        (df["birth_date"].dt.month == target_birth_date.month) &
        (df["birth_date"].dt.day == target_birth_date.day)
    ]

    # Exclude target player
    same_birthday = same_birthday[
        same_birthday["normalized_player"] !=
        player["normalized_player"]
    ]

    print("\nTARGET PLAYER BIRTHDAY:")
    print(target_birth_date.strftime("%B %d"))

    print("\nOTHER PLAYERS WITH THE SAME MONTH AND DAY:")

    if same_birthday.empty:

        print("No other players share this calendar birthday.")

    else:

        print(
            same_birthday[
                ["player", "birth_date", "pos", "colleges"]
            ].to_string(index=False)
        )

    # ======================================
    # Exact full date match
    # ======================================

    exact_date_matches = df[
        df["birth_date"] == target_birth_date
    ]

    exact_date_matches = exact_date_matches[
        exact_date_matches["normalized_player"] !=
        player["normalized_player"]
    ]

    print("\nPLAYERS WITH THE EXACT SAME FULL DATE OF BIRTH:")

    if exact_date_matches.empty:

        print("No exact full-date matches found.")

    else:

        print(
            exact_date_matches[
                ["player", "birth_date", "pos"]
            ].to_string(index=False)
        )

    return same_birthday


# ==========================================
# PHASE 5: EVALUATION AND SUMMARY
# ==========================================

def phase_5_evaluation_and_summary(df):

    print("\n[PHASE 5] Evaluation & Summary Reporting")
    print("-" * 65)

    total_players = len(df)

    valid_birth_dates = df["birth_date"].notna().sum()

    missing_birth_dates = df["birth_date"].isna().sum()

    # Career duration
    career_df = df.dropna(
        subset=["from", "to"]
    ).copy()

    career_df["career_duration_years"] = (
        career_df["to"] - career_df["from"] + 1
    )

    # Birthday frequency, ignoring year
    valid_dates = df.dropna(
        subset=["birth_date"]
    )

    birthday_counts = (
        valid_dates["birth_date"]
        .dt.strftime("%m-%d")
        .value_counts()
    )

    duplicate_birthday_dates = (
        birthday_counts[birthday_counts > 1]
    )

    players_with_shared_birthdays = int(
        birthday_counts[
            birthday_counts > 1
        ].sum()
    )

    print("\nDATASET COVERAGE:")

    print("Total Players:", total_players)
    print("Valid Birth Dates:", valid_birth_dates)
    print("Missing Birth Dates:", missing_birth_dates)

    print("\nBIRTHDAY SUMMARY:")

    print(
        "Distinct Calendar Birthdays:",
        birthday_counts.nunique()
        if False else birthday_counts.shape[0]
    )

    print(
        "Calendar Dates Shared by Multiple Players:",
        len(duplicate_birthday_dates)
    )

    print(
        "Players Belonging to Shared Birthday Groups:",
        players_with_shared_birthdays
    )

    print("\nCAREER LONGEVITY SUMMARY:")

    if not career_df.empty:

        print(
            "Average Career Duration:",
            round(
                career_df["career_duration_years"].mean(),
                2
            ),
            "years"
        )

        print(
            "Longest Career Duration:",
            career_df["career_duration_years"].max(),
            "years"
        )

        longest_career = career_df.loc[
            career_df["career_duration_years"].idxmax()
        ]

        print(
            "Player with Longest Career:",
            longest_career["player"]
        )

    print("\nMONTH WITH MOST BIRTHDAYS:")

    if not valid_dates.empty:

        month_counts = (
            valid_dates["birth_date"]
            .dt.month
            .value_counts()
        )

        most_common_month = month_counts.idxmax()

        print(
            datetime(2000, most_common_month, 1).strftime("%B"),
            "with",
            month_counts.max(),
            "players"
        )

    print("\nSUMMARY COMPLETED.")


# ==========================================
# FACADE FUNCTION: ORCHESTRATOR
# ==========================================

def run_analytics_lifecycle():

    print("=" * 65)

    print(
        "STARTING NBA BIRTHDAY & BIOGRAPHICAL LIFECYCLE"
    )

    print("=" * 65)

    target_player = TARGET_PLAYER

    # Phase 1
    df = phase_1_discovery_and_sanitization()

    # Phase 2
    valid_birthdays = phase_2_birthday_paradox_analysis(df)

    # Phase 3
    player = phase_3_player_lookup_system(
        df,
        target_player
    )

    # Phase 4
    if player is not None:

        phase_4_shared_birthday_cohort(
            df,
            target_player
        )

    # Phase 5
    phase_5_evaluation_and_summary(df)

    print("\n" + "=" * 65)

    print("NBA ANALYTICS LIFECYCLE COMPLETED SUCCESSFULLY")

    print("=" * 65)


# ==========================================
# MAIN GUARD
# ==========================================

if __name__ == "__main__":

    run_analytics_lifecycle()