# STEP 4 - ML PREPROCESSING
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from scipy.sparse import save_npz

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "ml_housing.csv"
)
MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)
PROCESSED_DIR = (
    PROJECT_ROOT
    / "Data"
    / "processed"
)
# Create folders if they do not exist
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)
PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# 2. LOAD ML DATASET

print("=" * 70)
print("REAL ESTATE INVESTMENT ADVISOR")
print("ML PREPROCESSING")
print("=" * 70)
print("\nLoading ML dataset...")

df = pd.read_csv(INPUT_FILE)
print("\nDataset loaded successfully.")

print(
    f"Dataset shape: {df.shape}"
)

# 3. CHECK TARGET COLUMNS
classification_target = "Good_Investment"
regression_target = "Future_Price_5Y"
required_targets = [
    classification_target,
    regression_target
]
print("\nChecking target columns...")
for target in required_targets:
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' is missing."
        )
    print(
        f"✓ {target}"
    )

# 4. SEPARATE TARGETS FROM FEATURES

print("\n" + "=" * 70)
print("SEPARATING FEATURES AND TARGETS")
print("=" * 70)

# Classification target

y_classification = df[
    classification_target
].copy()

# Regression target

y_regression = df[
    regression_target
].copy()

# Remove BOTH targets from feature dataset
X = df.drop(
    columns=[
        classification_target,
        regression_target,
        "ID"
    ],
    errors="ignore"
).copy()

print("\nClassification target:")
print("Good_Investment")

print("\nRegression target:")
print("Future_Price_5Y")

print("\nNumber of ML features before encoding:")

print(X.shape[1])

# 5. VERIFY PRICE_IN_LAKHS IS NOT PRESENT

print("\nChecking mentor-required exclusion...")

if "Price_in_Lakhs" in X.columns:
    raise ValueError(
        "ERROR: Price_in_Lakhs is still present in ML features."
    )
else:
    print(
        "✓ Price_in_Lakhs is NOT present in ML features."
    )

# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES

print("\n" + "=" * 70)
print("IDENTIFYING FEATURE TYPES")
print("=" * 70)

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
for column in numeric_features:
    print(
        "-",
        column
    )
print("\nCategorical features:")
for column in categorical_features:
    print(
        "-",
        column
    )

# 7. CHECK MISSING VALUES

print("\n" + "=" * 70)
print("CHECKING MISSING VALUES")
print("=" * 70)
missing_values = X.isnull().sum().sum()
print(
    f"Total missing values: {missing_values}"
)
if missing_values > 0:
    print(
        "\nWARNING: Missing values detected."
    )
else:
    print(
        "✓ No missing values found."
    )

# 8. CREATE PREPROCESSING PIPELINE

print("\n" + "=" * 70)
print("CREATING PREPROCESSING PIPELINE")
print("=" * 70)

# Numerical features:
# StandardScaler is used for normalization.

numeric_transformer = StandardScaler()

# Categorical features:
# OneHotEncoder converts categorical values into numbers.

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)
print(
    "\n✓ Numerical features will be standardized."
)

print(
    "✓ Categorical features will be One-Hot Encoded."
)

# 9. CLASSIFICATION TRAIN / TEST SPLIT

print("\n" + "=" * 70)
print("CLASSIFICATION TRAIN / TEST SPLIT")
print("=" * 70)

X_train_class, X_test_class, y_train_class, y_test_class = (
    train_test_split(
        X,
        y_classification,
        test_size=0.20,
        random_state=42,
        stratify=y_classification
    )
)
print("\nClassification data:")

print(
    f"Training samples: {X_train_class.shape[0]}"
)

print(
    f"Testing samples:  {X_test_class.shape[0]}"
)

# 10. FIT CLASSIFICATION PREPROCESSOR

print("\nFitting classification preprocessing...")
classification_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numeric_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_features
        )
    ]
)
X_train_class_processed = (
    classification_preprocessor.fit_transform(
        X_train_class
    )
)
X_test_class_processed = (
    classification_preprocessor.transform(
        X_test_class
    )
)
print(
    "\nClassification preprocessing completed."
)
print(
    "Processed training shape:",
    X_train_class_processed.shape
)
print(
    "Processed testing shape:",
    X_test_class_processed.shape
)

# 11. SAVE CLASSIFICATION DATA
save_npz(
    PROCESSED_DIR
    / "X_train_classification.npz",
    X_train_class_processed
)
save_npz(
    PROCESSED_DIR
    / "X_test_classification.npz",
    X_test_class_processed
)
y_train_class.to_csv(
    PROCESSED_DIR
    / "y_train_classification.csv",
    index=False
)
y_test_class.to_csv(
    PROCESSED_DIR
    / "y_test_classification.csv",
    index=False
)
joblib.dump(
    classification_preprocessor,
    MODEL_DIR
    / "classification_preprocessor.pkl"
)
print(
    "\n✓ Classification preprocessing files saved."
)

# 12. REGRESSION TRAIN / TEST SPLIT

print("\n" + "=" * 70)
print("REGRESSION TRAIN / TEST SPLIT")
print("=" * 70)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
    train_test_split(
        X,
        y_regression,
        test_size=0.20,
        random_state=42
    )
)
print("\nRegression data:")

print(
    f"Training samples: {X_train_reg.shape[0]}"
)
print(
    f"Testing samples:  {X_test_reg.shape[0]}"
)

# 13. FIT REGRESSION PREPROCESSOR

print("\nFitting regression preprocessing...")
regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numeric_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_features
        )
    ]
)
X_train_reg_processed = (
    regression_preprocessor.fit_transform(
        X_train_reg
    )
)
X_test_reg_processed = (
    regression_preprocessor.transform(
        X_test_reg
    )
)
print(
    "\nRegression preprocessing completed."
)
print(
    "Processed training shape:",
    X_train_reg_processed.shape
)
print(
    "Processed testing shape:",
    X_test_reg_processed.shape
)

# 14. SAVE REGRESSION DATA

save_npz(
    PROCESSED_DIR
    / "X_train_regression.npz",
    X_train_reg_processed
)
save_npz(
    PROCESSED_DIR
    / "X_test_regression.npz",
    X_test_reg_processed
)
y_train_reg.to_csv(
    PROCESSED_DIR
    / "y_train_regression.csv",
    index=False
)
y_test_reg.to_csv(
    PROCESSED_DIR
    / "y_test_regression.csv",
    index=False
)
joblib.dump(
    regression_preprocessor,
    MODEL_DIR
    / "regression_preprocessor.pkl"
)
print(
    "\n✓ Regression preprocessing files saved."
)

# 15. FINAL SUMMARY

print("\n" + "=" * 70)
print("ML PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nClassification:")

print(
    "Target: Good_Investment"
)
print(
    f"Training rows: {X_train_class.shape[0]}"
)
print(
    f"Testing rows: {X_test_class.shape[0]}"
)
print(
    f"Encoded feature count: "
    f"{X_train_class_processed.shape[1]}"
)
print("\nRegression:")

print(
    "Target: Future_Price_5Y"
)
print(
    f"Training rows: {X_train_reg.shape[0]}"
)
print(
    f"Testing rows: {X_test_reg.shape[0]}"
)
print(
    f"Encoded feature count: "
    f"{X_train_reg_processed.shape[1]}"
)
print("\nSaved preprocessing objects:")
print(
    "- models/classification_preprocessor.pkl"
)
print(
    "- models/regression_preprocessor.pkl"
)
print("\nSaved processed datasets:")

print(
    "- X_train_classification.npz"
)
print(
    "- X_test_classification.npz"
)
print(
    "- y_train_classification.csv"
)
print(
    "- y_test_classification.csv"
)
print(
    "- X_train_regression.npz"
)
print(
    "- X_test_regression.npz"
)
print(
    "- y_train_regression.csv"
)
print(
    "- y_test_regression.csv"
)
print("\n" + "=" * 70)
print("READY FOR MODEL TRAINING")
print("=" * 70)