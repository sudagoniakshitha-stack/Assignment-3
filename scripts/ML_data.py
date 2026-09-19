
# Purpose:
# 1. Load cleaned housing dataset
# 2. Create investment-related features
# 3. Create Good_Investment target
# 4. Create Future_Price_5Y target
# 5. Remove Price_in_Lakhs from ML features
# 6. Save final ML dataset
# 1. IMPORT LIBRARIES
import os
import numpy as np
import pandas as pd

# 2. DEFINE PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "cleaned_housing.csv"
)
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "ml_housing.csv"
)
# 3. LOAD CLEANED DATASET
print("=" * 70)
print("LOADING CLEANED DATASET")
print("=" * 70)
df = pd.read_csv(INPUT_FILE)
print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# 4. DISPLAY ORIGINAL COLUMNS

print("\nOriginal columns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# 5. CHECK REQUIRED COLUMNS

required_columns = [
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Size_in_SqFt",
    "BHK",
    "Age_of_Property",
    "Parking_Space",
    "Security",
    "Availability_Status",
    "Public_Transport_Accessibility",
    "Amenity_Count"
]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]
if missing_columns:
    print("\nERROR: Required columns are missing:")
    for column in missing_columns:
        print("-", column)
    raise ValueError(
        "Required columns are missing from cleaned_housing.csv"
    )
print("\nAll required columns are available.")

# 6. ENSURE NUMERICAL COLUMNS ARE NUMERIC

numeric_columns = [
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Size_in_SqFt",
    "BHK",
    "Age_of_Property",
    "Amenity_Count"
]
for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )
# Remove rows with missing values created by conversion
df = df.dropna(
    subset=numeric_columns
).reset_index(drop=True)
print(
    f"\nDataset after numeric validation: {df.shape}"
)

# 7. PRICE PER SQ FT

median_price_per_sqft = df["Price_per_SqFt"].median()

df["Price_per_SqFt_Score"] = np.where(
    df["Price_per_SqFt"] <= median_price_per_sqft,
    1,
    0
)
print("\nPrice per SqFt criterion created.")

print(
    f"Median Price per SqFt: "
    f"{median_price_per_sqft:.4f}"
)

# 8. BHK SCORE

# PDF multi-factor example:
# BHK >= 3 -> positive investment factor

df["BHK_Score"] = np.where(
    df["BHK"] >= 3,
    1,
    0
)
print("BHK score created.")

# 9. AVAILABILITY SCORE
# Ready/available properties are preferable for the
# investment scoring approach.

# We use the actual Availability_Status values in the
# dataset rather than inventing new categories.

df["Availability_Score"] = np.where(
    df["Availability_Status"].astype(str).str.strip().str.lower()
    == "ready_to_move",
    1,
    0
)
print("Availability score created.")

# 10. PUBLIC TRANSPORT SCORE
# High accessibility -> positive factor

df["Transport_Score"] = np.where(
    df["Public_Transport_Accessibility"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "high",
    1,
    0
)
print("Public transport score created.")

# 11. PARKING SCORE
# Parking available -> positive factor
df["Parking_Score"] = np.where(
    df["Parking_Space"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["yes", "available"]),
    1,
    0
)
print("Parking score created.")

# 12. SECURITY SCORE
# If security information is present, give positive score.
df["Security_Score"] = np.where(
    df["Security"].astype(str).str.strip().str.lower()
    .isin(["yes", "high", "gated", "cctv", "guard"]),
    1,
    0
)
print("Security score created.")

# 13. AMENITY SCORE
# Properties with at least the median number of amenities
# receive a positive score.
median_amenity_count = df["Amenity_Count"].median()
df["Amenity_Score"] = np.where(
    df["Amenity_Count"] >= median_amenity_count,
    1,
    0
)
print(
    f"Amenity score created. "
    f"Median amenities: {median_amenity_count:.2f}"
)

# 14. PROPERTY AGE SCORE

# Newer properties receive a positive factor.
# Properties with age <= median age receive score 1.

median_age = df["Age_of_Property"].median()
df["Age_Score"] = np.where(
    df["Age_of_Property"] <= median_age,
    1,
    0
)
print(
    f"Age score created. "
    f"Median property age: {median_age:.2f}"
)

# 15. CREATE GOOD_INVESTMENT TARGET
# PDF Method 1:
# Price_in_Lakhs <= median price
#       -> Good Investment = 1
# Price_in_Lakhs > median price
#       -> Good Investment = 0
# Price_in_Lakhs is used only to construct the target.
# It will be removed from the ML feature dataset later.

median_price = df["Price_in_Lakhs"].median()
df["Good_Investment"] = np.where(
    df["Price_in_Lakhs"] <= median_price,
    1,
    0
)
print("\nGood_Investment target created.")
print(
    f"Median Price in Lakhs: "
    f"{median_price:.2f}"
)
print("\nGood_Investment distribution:")
print(
    df["Good_Investment"]
    .value_counts()
    .sort_index()
)
print("\nTarget percentages:")
print(
    df["Good_Investment"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)
# 16. CREATE FUTURE PRICE TARGET
# PDF formula:
# Future = Current * (1 + r)^t
# Example:
# r = 8%
# t = 5 years
# Therefore:
# Future_Price_5Y =
# Price_in_Lakhs * (1 + 0.08)^5
growth_rate = 0.08
years = 5

df["Future_Price_5Y"] = (
    df["Price_in_Lakhs"]
    * (1 + growth_rate) ** years
)
print("\nFuture_Price_5Y target created.")
print(
    f"Growth rate used: "
    f"{growth_rate * 100:.0f}%"
)
print(
    f"Forecast period: "
    f"{years} years"
)
print(
    "Formula: Future Price = "
    "Current Price * (1 + 0.08)^5"
)

# 17. CALCULATE EXPECTED APPRECIATION
appreciation = (
    ((1 + growth_rate) ** years) - 1
) * 100
df["Expected_Appreciation_Percentage"] = appreciation

print(
    f"\nExpected 5-year appreciation: "
    f"{appreciation:.2f}%"
)
# 18. IMPORTANT:
# DROP PRICE_IN_LAKHS FROM ML FEATURES

# Mentor instruction:
#
# Price_in_Lakhs must NOT be used as an ML feature.
# We keep the original column temporarily because it is
# needed to create Good_Investment and Future_Price_5Y.
# After target creation, it is removed.
print(
    "\nChecking Price_in_Lakhs before removal:"
)
if "Price_in_Lakhs" in df.columns:

    print("Price_in_Lakhs is present.")
else:
    print(
        "Price_in_Lakhs is missing unexpectedly."
    )

# 19. TARGET-CONSTRUCTION HELPER COLUMNS
# The score columns created above are retained as engineered
# features. They are not used to construct Good_Investment
# because the selected PDF Method 1 uses median price.

print(
    "\nInvestment-related engineered features retained."
)

# 20. DROP PRICE_IN_LAKHS FROM ML DATA
df_ml = df.drop(
    columns=["Price_in_Lakhs"],
    errors="ignore"
).copy()
print(
    "\nPrice_in_Lakhs removed from ML dataset."
)

# 21. CHECK THAT PRICE_IN_LAKHS IS REALLY REMOVED

if "Price_in_Lakhs" in df_ml.columns:

    raise ValueError(
        "ERROR: Price_in_Lakhs is still present "
        "in ML dataset!"
    )
else:
    print(
        "SUCCESS: Price_in_Lakhs is NOT present "
        "in ML features."
    )

# 22. TARGET COLUMNS

classification_target = "Good_Investment"
regression_target = "Future_Price_5Y"
print("\nTargets:")
print(
    f"1. Classification: "
    f"{classification_target}"
)
print(
    f"2. Regression: "
    f"{regression_target}"
)

# 23. DISPLAY FINAL ML DATASET INFORMATION

print("\n" + "=" * 70)
print("FINAL ML DATASET")
print("=" * 70)
print(
    f"\nFinal dataset shape: "
    f"{df_ml.shape}"
)
print("\nFinal ML dataset columns:")
for i, column in enumerate(
    df_ml.columns,
    start=1
):
    print(
        f"{i}. {column}"
    )
# 24. TARGET VALIDATION

print("\n" + "=" * 70)
print("TARGET VALIDATION")
print("=" * 70)
print("\nGood_Investment counts:")
print(
    df_ml["Good_Investment"]
    .value_counts()
    .sort_index()
)
print("\nGood_Investment percentages:")
print(
    df_ml["Good_Investment"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)
print("\nFuture_Price_5Y statistics:")

print(
    df_ml["Future_Price_5Y"]
    .describe()
)

# 25. CHECK MISSING VALUES

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)
missing_values = (
    df_ml.isnull().sum().sum()
)
duplicate_rows = (
    df_ml.duplicated().sum()
)
print(
    f"\nTotal missing values: "
    f"{missing_values}"
)
print(
    f"Total duplicate rows: "
    f"{duplicate_rows}"
)

# 26. SAVE ML DATASET

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)
df_ml.to_csv(
    OUTPUT_FILE,
    index=False
)

# 27. FINAL MESSAGE

print("\n" + "=" * 70)
print("ML DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nSaved file:")
print(OUTPUT_FILE)
print("\nTargets created:")
print("1. Good_Investment")
print("2. Future_Price_5Y")
print("\nImportant:")
print(
    "Price_in_Lakhs was used for target construction "
    "but was DROPPED before ML feature creation."
)
print("\nClassification methodology:")
print(
    "PDF Method 1: "
    "Price_in_Lakhs <= median price "
    "-> Good Investment"
)
print("\nRegression formula:")
print(
    "Future Price = "
    "Current Price * (1 + 0.08)^5"
)
print(
    f"\nExpected 5-year appreciation: "
    f"{appreciation:.2f}%"
)
print("\n" + "=" * 70)