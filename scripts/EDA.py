# EXPLORATORY DATA ANALYSIS - 20 QUESTIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD CLEANED DATASET
DATA_PATH = "Data/processed/cleaned_housing.csv"
df = pd.read_csv(DATA_PATH)
print("\n" + "=" * 70)
print("REAL ESTATE INVESTMENT ADVISOR - EDA")
print("=" * 70)
print("\nDataset shape:")
print(df.shape)
print("\nColumns:")
print(df.columns.tolist())

# BASIC DATA INFORMATION

print("\n" + "=" * 70)
print("BASIC DATA INFORMATION")
print("=" * 70)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes)

# MAKE SURE NUMERICAL COLUMNS ARE NUMERIC

numeric_columns = [
    "Price_in_Lakhs",
    "Size_in_SqFt",
    "Price_per_SqFt",
    "Year_Built",
    "Age_of_Property",
    "Floor_No",
    "Total_Floors",
    "BHK",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Amenity_Count"
]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 1.Distribution of property prices

print("\n" + "=" * 70)
print("QUESTION 1: DISTRIBUTION OF PROPERTY PRICES")
print("=" * 70)

print(df["Price_in_Lakhs"].describe())

plt.figure(figsize=(10, 6))
plt.hist(df["Price_in_Lakhs"], bins=40, edgecolor="black")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Number of Properties")
plt.title("Q1 - Distribution of Property Prices")
plt.grid(axis="y", alpha=0.3)
plt.show()

#2. Distribution of property sizes

print("\n" + "=" * 70)
print("QUESTION 2: DISTRIBUTION OF PROPERTY SIZES")
print("=" * 70)

print(df["Size_in_SqFt"].describe())

plt.figure(figsize=(10, 6))
plt.hist(df["Size_in_SqFt"], bins=40, edgecolor="black")
plt.xlabel("Size (Sq Ft)")
plt.ylabel("Number of Properties")
plt.title("Q2 - Distribution of Property Sizes")
plt.grid(axis="y", alpha=0.3)
plt.show()

#3.Price per sq ft by property type

print("\n" + "=" * 70)
print("QUESTION 3: PRICE PER SQ FT BY PROPERTY TYPE")
print("=" * 70)

q3 = (
    df.groupby("Property_Type")["Price_per_SqFt"]
    .mean()
    .sort_values(ascending=False)
)
print(q3)
plt.figure(figsize=(10, 6))
q3.plot(kind="bar", edgecolor="black")
plt.xlabel("Property Type")
plt.ylabel("Average Price per Sq Ft")
plt.title("Q3 - Average Price per Sq Ft by Property Type")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

#4. Relationship between property size and price

print("\n" + "=" * 70)
print("QUESTION 4: RELATIONSHIP BETWEEN SIZE AND PRICE")
print("=" * 70)
correlation_size_price = df["Size_in_SqFt"].corr(
    df["Price_in_Lakhs"]
)
print(
    f"Correlation between Size and Price: "
    f"{correlation_size_price:.4f}"
)
plt.figure(figsize=(10, 6))
plt.scatter(
    df["Size_in_SqFt"],
    df["Price_in_Lakhs"],
    alpha=0.3
)
plt.xlabel("Size (Sq Ft)")
plt.ylabel("Price (Lakhs)")
plt.title("Q4 - Property Size vs Property Price")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5.Outliers in price per sq ft or property size

print("\n" + "=" * 70)
print("QUESTION 5: OUTLIERS IN PRICE PER SQ FT AND PROPERTY SIZE")
print("=" * 70)

def find_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[
        (series < lower) | (series > upper)
    ]
    return outliers, lower, upper
size_outliers, size_lower, size_upper = find_outliers_iqr(
    df["Size_in_SqFt"]
)
ppsft_outliers, ppsft_lower, ppsft_upper = find_outliers_iqr(
    df["Price_per_SqFt"]
)

print(f"Size lower bound: {size_lower:.2f}")
print(f"Size upper bound: {size_upper:.2f}")
print(f"Number of Size outliers: {len(size_outliers)}")

print(f"\nPrice/SqFt lower bound: {ppsft_lower:.4f}")
print(f"Price/SqFt upper bound: {ppsft_upper:.4f}")
print(f"Number of Price/SqFt outliers: {len(ppsft_outliers)}")

plt.figure(figsize=(10, 6))
plt.boxplot(
    [
        df["Size_in_SqFt"].dropna(),
        df["Price_per_SqFt"].dropna()
    ],
    tick_labels=["Size (Sq Ft)", "Price per Sq Ft"]
)
plt.title("Q5 - Outlier Detection")
plt.ylabel("Value")
plt.grid(axis="y", alpha=0.3)
plt.show()

# 6. Average price per sq ft by state

print("\n" + "=" * 70)
print("QUESTION 6: AVERAGE PRICE PER SQ FT BY STATE")
print("=" * 70)
q6 = (
    df.groupby("State")["Price_per_SqFt"]
    .mean()
    .sort_values(ascending=False)
)
print(q6)

plt.figure(figsize=(14, 7))
q6.plot(kind="bar", edgecolor="black")
plt.xlabel("State")
plt.ylabel("Average Price per Sq Ft")
plt.title("Q6 - Average Price per Sq Ft by State")
plt.xticks(rotation=75)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 7.Average property price by city

print("\n" + "=" * 70)
print("QUESTION 7: AVERAGE PROPERTY PRICE BY CITY")
print("=" * 70)
q7 = (
    df.groupby("City")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
)
print(q7)

plt.figure(figsize=(14, 7))
q7.plot(kind="bar", edgecolor="black")
plt.xlabel("City")
plt.ylabel("Average Price (Lakhs)")
plt.title("Q7 - Average Property Price by City")
plt.xticks(rotation=75)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 8. Median age of properties by locality

print("\n" + "=" * 70)
print("QUESTION 8: MEDIAN AGE OF PROPERTIES BY LOCALITY")
print("=" * 70)
q8 = (
    df.groupby("Locality")["Age_of_Property"]
    .median()
    .sort_values(ascending=False)
)
print("\nTop 20 localities by median property age:")
print(q8.head(20))

plt.figure(figsize=(14, 7))
q8.head(20).sort_values().plot(
    kind="barh",
    edgecolor="black"
)
plt.xlabel("Median Property Age")
plt.ylabel("Locality")
plt.title("Q8 - Top 20 Localities by Median Property Age")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# 9.BHK distribution across cities

print("\n" + "=" * 70)
print("QUESTION 9: BHK DISTRIBUTION ACROSS CITIES")
print("=" * 70)
q9 = pd.crosstab(
    df["City"],
    df["BHK"]
)
print(q9)

plt.figure(figsize=(14, 8))
q9.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 8)
)
plt.xlabel("City")
plt.ylabel("Number of Properties")
plt.title("Q9 - BHK Distribution Across Cities")
plt.xticks(rotation=75)
plt.legend(title="BHK")
plt.tight_layout()
plt.show()

# 10. Price trends / price comparison for top 5 expensive localities

print("\n" + "=" * 70)
print("QUESTION 10: TOP 5 MOST EXPENSIVE LOCALITIES")
print("=" * 70)
q10 = (
    df.groupby("Locality")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
)
top5_localities = q10.head(5)

print("\nTop 5 most expensive localities:")
print(top5_localities)

plt.figure(figsize=(10, 6))
top5_localities.sort_values().plot(
    kind="barh",
    edgecolor="black"
)
plt.xlabel("Average Property Price (Lakhs)")
plt.ylabel("Locality")
plt.title("Q10 - Top 5 Most Expensive Localities")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

print(
    "\nNOTE: The provided dataset does not contain a date/time column."
)
print(
    "Therefore, a genuine historical 'price trend over time' "
    "cannot be calculated from this dataset."
)
print(
    "This analysis compares the average prices of the top 5 "
    "most expensive localities instead of inventing a time trend."
)

#11. Correlation between numeric features

print("\n" + "=" * 70)
print("QUESTION 11: NUMERIC FEATURE CORRELATION")
print("=" * 70)

correlation_columns = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Age_of_Property",
    "Floor_No",
    "Total_Floors",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Amenity_Count"
]
correlation_columns = [
    col for col in correlation_columns
    if col in df.columns
]
corr_matrix = df[correlation_columns].corr()

print(corr_matrix)

plt.figure(figsize=(12, 9))
plt.imshow(
    corr_matrix,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=75
)
plt.yticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns
)
plt.title("Q11 - Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()

#12. Nearby schools vs price per sq ft

print("\n" + "=" * 70)
print("QUESTION 12: NEARBY SCHOOLS VS PRICE PER SQ FT")
print("=" * 70)
q12 = (
    df.groupby("Nearby_Schools")["Price_per_SqFt"]
    .mean()
    .sort_index()
)
print(q12)

plt.figure(figsize=(10, 6))
plt.scatter(
    df["Nearby_Schools"],
    df["Price_per_SqFt"],
    alpha=0.3
)
plt.xlabel("Nearby Schools")
plt.ylabel("Price per Sq Ft")
plt.title("Q12 - Nearby Schools vs Price per Sq Ft")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(
    f"Correlation: "
    f"{df['Nearby_Schools'].corr(df['Price_per_SqFt']):.4f}"
)

# 13. Nearby hospitals vs price per sq ft

print("\n" + "=" * 70)
print("QUESTION 13: NEARBY HOSPITALS VS PRICE PER SQ FT")
print("=" * 70)
q13 = (
    df.groupby("Nearby_Hospitals")["Price_per_SqFt"]
    .mean()
    .sort_index()
)
print(q13)

plt.figure(figsize=(10, 6))
plt.scatter(
    df["Nearby_Hospitals"],
    df["Price_per_SqFt"],
    alpha=0.3
)
plt.xlabel("Nearby Hospitals")
plt.ylabel("Price per Sq Ft")
plt.title("Q13 - Nearby Hospitals vs Price per Sq Ft")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(
    f"Correlation: "
    f"{df['Nearby_Hospitals'].corr(df['Price_per_SqFt']):.4f}"
)

# 14.Price by furnished status

print("\n" + "=" * 70)
print("QUESTION 14: PRICE BY FURNISHED STATUS")
print("=" * 70)
q14 = (
    df.groupby("Furnished_Status")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
)
print(q14)

plt.figure(figsize=(10, 6))
q14.plot(kind="bar", edgecolor="black")
plt.xlabel("Furnished Status")
plt.ylabel("Average Price (Lakhs)")
plt.title("Q14 - Average Price by Furnished Status")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 15.Price per sq ft by facing direction

print("\n" + "=" * 70)
print("QUESTION 15: PRICE PER SQ FT BY FACING")
print("=" * 70)
q15 = (
    df.groupby("Facing")["Price_per_SqFt"]
    .mean()
    .sort_values(ascending=False)
)
print(q15)

plt.figure(figsize=(10, 6))
q15.plot(kind="bar", edgecolor="black")
plt.xlabel("Facing Direction")
plt.ylabel("Average Price per Sq Ft")
plt.title("Q15 - Price per Sq Ft by Facing Direction")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 16.Number of properties by owner type

print("\n" + "=" * 70)
print("QUESTION 16: PROPERTIES BY OWNER TYPE")
print("=" * 70)
q16 = df["Owner_Type"].value_counts()

print(q16)

plt.figure(figsize=(10, 6))
q16.plot(kind="bar", edgecolor="black")
plt.xlabel("Owner Type")
plt.ylabel("Number of Properties")
plt.title("Q16 - Number of Properties by Owner Type")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 17.Number of properties by availability status

print("\n" + "=" * 70)
print("QUESTION 17: PROPERTIES BY AVAILABILITY STATUS")
print("=" * 70)
q17 = df["Availability_Status"].value_counts()
print(q17)
plt.figure(figsize=(10, 6))
q17.plot(kind="bar", edgecolor="black")
plt.xlabel("Availability Status")
plt.ylabel("Number of Properties")
plt.title("Q17 - Properties by Availability Status")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 18.Does parking space affect property price?

print("\n" + "=" * 70)
print("QUESTION 18: PARKING SPACE VS PROPERTY PRICE")
print("=" * 70)
q18 = (
    df.groupby("Parking_Space")["Price_in_Lakhs"]
    .mean()
    .sort_index()
)
print(q18)

plt.figure(figsize=(10, 6))
q18.plot(kind="bar", edgecolor="black")
plt.xlabel("Parking Space")
plt.ylabel("Average Property Price (Lakhs)")
plt.title("Q18 - Parking Space vs Property Price")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 19.Amenities vs price per sq ft

print("\n" + "=" * 70)
print("QUESTION 19: AMENITIES VS PRICE PER SQ FT")
print("=" * 70)
q19 = (
    df.groupby("Amenity_Count")["Price_per_SqFt"]
    .mean()
    .sort_index()
)
print(q19)

plt.figure(figsize=(10, 6))
plt.scatter(
    df["Amenity_Count"],
    df["Price_per_SqFt"],
    alpha=0.3
)
plt.xlabel("Number of Amenities")
plt.ylabel("Price per Sq Ft")
plt.title("Q19 - Amenities vs Price per Sq Ft")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(
    f"Correlation between Amenity Count and Price per Sq Ft: "
    f"{df['Amenity_Count'].corr(df['Price_per_SqFt']):.4f}"
)

# 20.Public transport accessibility vs price per sq ft

print("\n" + "=" * 70)
print("QUESTION 20: PUBLIC TRANSPORT VS PRICE PER SQ FT")
print("=" * 70)
q20 = (
    df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"]
    .mean()
    .sort_values(ascending=False)
)
print(q20)
plt.figure(figsize=(10, 6))
q20.plot(kind="bar", edgecolor="black")
plt.xlabel("Public Transport Accessibility")
plt.ylabel("Average Price per Sq Ft")
plt.title("Q20 - Public Transport Accessibility vs Price per Sq Ft")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# FINAL EDA SUMMARY

print("\n" + "=" * 70)
print("EDA QUESTIONS 1-20 COMPLETED")
print("=" * 70)
print("\nTotal records analysed:", len(df))

print("\nAverage Property Price:")
print(f"{df['Price_in_Lakhs'].mean():.2f} Lakhs")

print("\nMedian Property Price:")
print(f"{df['Price_in_Lakhs'].median():.2f} Lakhs")

print("\nAverage Property Size:")
print(f"{df['Size_in_SqFt'].mean():.2f} SqFt")

print("\nMedian Property Size:")
print(f"{df['Size_in_SqFt'].median():.2f} SqFt")

print("\nAverage Price per SqFt:")
print(f"{df['Price_per_SqFt'].mean():.4f}")

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)