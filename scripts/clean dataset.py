# ============================================================
# REAL ESTATE INVESTMENT ADVISOR
# STEP 1 - DATA CLEANING
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "Data" / "raw" / "india_housing_prices.csv"

OUTPUT_FOLDER = PROJECT_ROOT / "Data" / "processed"

OUTPUT_FILE = OUTPUT_FOLDER / "cleaned_housing.csv"


# Create processed folder if required
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")


# ------------------------------------------------------------
# 3. BASIC INFORMATION
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 4. CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = df.columns.str.strip()


# ------------------------------------------------------------
# 5. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values before cleaning:")

print(df.isnull().sum())


# ------------------------------------------------------------
# 6. HANDLE MISSING VALUES
# ------------------------------------------------------------

# Numerical columns
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(median_value)


# Categorical columns
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()[0]

        df[column] = df[column].fillna(mode_value)


# ------------------------------------------------------------
# 7. REMOVE DUPLICATES
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows found:", duplicate_count)

df = df.drop_duplicates()

print("Duplicate rows after cleaning:", df.duplicated().sum())


# ------------------------------------------------------------
# 8. CLEAN TEXT VALUES
# ------------------------------------------------------------

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:

    df[column] = df[column].astype(str).str.strip()


# ------------------------------------------------------------
# 9. CHECK SIZE
# ------------------------------------------------------------

invalid_size = df["Size_in_SqFt"] <= 0

print(
    "\nInvalid Size_in_SqFt values:",
    invalid_size.sum()
)

if invalid_size.sum() > 0:

    median_size = df.loc[
        ~invalid_size,
        "Size_in_SqFt"
    ].median()

    df.loc[
        invalid_size,
        "Size_in_SqFt"
    ] = median_size


# ------------------------------------------------------------
# 10. CHECK PRICE
# ------------------------------------------------------------

invalid_price = df["Price_in_Lakhs"] <= 0

print(
    "Invalid Price_in_Lakhs values:",
    invalid_price.sum()
)

if invalid_price.sum() > 0:

    median_price = df.loc[
        ~invalid_price,
        "Price_in_Lakhs"
    ].median()

    df.loc[
        invalid_price,
        "Price_in_Lakhs"
    ] = median_price


# ------------------------------------------------------------
# 11. CHECK YEAR BUILT
# ------------------------------------------------------------

invalid_year = (
    (df["Year_Built"] < 1900)
    |
    (df["Year_Built"] > 2025)
)

print(
    "Invalid Year_Built values:",
    invalid_year.sum()
)

if invalid_year.sum() > 0:

    median_year = df.loc[
        ~invalid_year,
        "Year_Built"
    ].median()

    df.loc[
        invalid_year,
        "Year_Built"
    ] = median_year


# ------------------------------------------------------------
# 12. CHECK TOTAL FLOORS
# ------------------------------------------------------------

invalid_total_floors = df["Total_Floors"] <= 0

print(
    "Invalid Total_Floors values:",
    invalid_total_floors.sum()
)

if invalid_total_floors.sum() > 0:

    median_floors = df.loc[
        ~invalid_total_floors,
        "Total_Floors"
    ].median()

    df.loc[
        invalid_total_floors,
        "Total_Floors"
    ] = median_floors


# ------------------------------------------------------------
# 13. CHECK FLOOR NUMBER
# ------------------------------------------------------------

invalid_floor = df["Floor_No"] < 0

print(
    "Invalid Floor_No values:",
    invalid_floor.sum()
)

if invalid_floor.sum() > 0:

    df.loc[
        invalid_floor,
        "Floor_No"
    ] = 0


# ------------------------------------------------------------
# 14. FLOOR VALIDATION
# ------------------------------------------------------------

invalid_floor_relation = (
    df["Floor_No"] > df["Total_Floors"]
)

print(
    "Floor_No > Total_Floors:",
    invalid_floor_relation.sum()
)

df.loc[
    invalid_floor_relation,
    "Floor_No"
] = df.loc[
    invalid_floor_relation,
    "Total_Floors"
]


# ------------------------------------------------------------
# 15. AGE OF PROPERTY
# ------------------------------------------------------------

# The supplied dataset uses 2025 as the reference year.

df["Age_of_Property"] = (
    2025 - df["Year_Built"]
)

print(
    "\nAge_of_Property calculated successfully."
)


# ------------------------------------------------------------
# 16. PRICE PER SQFT
# ------------------------------------------------------------

df["Price_per_SqFt"] = (
    df["Price_in_Lakhs"]
    /
    df["Size_in_SqFt"]
)

print(
    "Price_per_SqFt calculated successfully."
)


# ------------------------------------------------------------
# 17. AMENITY COUNT
# ------------------------------------------------------------

def count_amenities(value):

    if pd.isna(value):
        return 0

    amenities = str(value).split(",")

    amenities = [
        item.strip()
        for item in amenities
        if item.strip() != ""
    ]

    return len(set(amenities))


df["Amenity_Count"] = (
    df["Amenities"].apply(count_amenities)
)

print(
    "Amenity_Count created successfully."
)


# ------------------------------------------------------------
# 18. FINAL CHECK
# ------------------------------------------------------------

print("\nFinal missing values:")

print(
    df.isnull().sum().sum()
)

print(
    "\nFinal duplicate rows:",
    df.duplicated().sum()
)

print(
    "\nFinal dataset shape:",
    df.shape
)


# ------------------------------------------------------------
# 19. SAVE CLEANED DATASET
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 20. SUCCESS MESSAGE
# ------------------------------------------------------------

print("\n==========================================")
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("==========================================")

print(
    "\nCleaned dataset saved at:"
)

print(OUTPUT_FILE)
