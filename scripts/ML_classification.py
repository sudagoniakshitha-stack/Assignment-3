import os
import warnings
import joblib
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# 1. PROJECT PATHS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "ml_housing.csv"
)
MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. LOAD ML DATASET

print("\n" + "=" * 70)
print("REAL ESTATE INVESTMENT ADVISOR")
print("CLASSIFICATION MODEL DEVELOPMENT")
print("=" * 70)
print("\nLoading ML dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# 3. CHECK TARGET

if "Good_Investment" not in df.columns:
    raise ValueError(
        "Good_Investment target column is missing from ml_housing.csv"
    )
print("\nTarget column found: Good_Investment")

print("\nTarget distribution:")
print(df["Good_Investment"].value_counts())

print("\nTarget percentage:")
print(
    df["Good_Investment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)
# 4. REMOVE TARGET COLUMNS
# Good_Investment is the classification target.
#
# Future_Price_5Y is another target and must NOT be used
# as an input feature.
#
# Price_in_Lakhs was specifically instructed by the mentor
# to be dropped during ML.
#
# Property_ID / ID are identifiers and are not useful
# predictive features.

DROP_COLUMNS = [
    "Good_Investment",
    "Future_Price_5Y",
    "Price_in_Lakhs",
    "Property_ID",
    "ID"
]
drop_existing = [
    col for col in DROP_COLUMNS
    if col in df.columns
]
X = df.drop(columns=drop_existing)
y = df["Good_Investment"]
print("\nColumns removed from ML features:")
for col in drop_existing:
    print(f" - {col}")

# 5. CHECK FEATURES

print("\nFinal feature columns before encoding:")
print("-" * 70)

for i, col in enumerate(X.columns, start=1):
    print(f"{i}. {col}")

print(f"\nTotal raw features: {X.shape[1]}")

# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

# 7. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")

# 8. PREPROCESSING

print("\nCreating preprocessing pipeline...")
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)
categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)

# 9. FIT PREPROCESSOR

print("Fitting preprocessing pipeline...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("Preprocessing completed.")
print(
    f"Processed training shape: {X_train_processed.shape}"
)
print(
    f"Processed testing shape : {X_test_processed.shape}"
)

# 10. SAVE PREPROCESSOR

preprocessor_path = os.path.join(
    MODEL_DIR,
    "classification_preprocessor.pkl"
)
joblib.dump(
    preprocessor,
    preprocessor_path
)
print(
    f"\nPreprocessor saved to:\n{preprocessor_path}"
)

# 11. DEFINE 5 CLASSIFICATION MODELS

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=20
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        n_jobs=-1,
        max_depth=20
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=10,
        random_state=42,
        n_jobs=-1,
        max_depth=20
    )
}

# 12. ADD XGBOOST

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
else:
    print("\nWARNING:")
    print("XGBoost is not installed.")
    print("Install it using:")
    print("pip install xgboost")

# 13. MODEL TRAINING

results = []
trained_models = {}

print("\n" + "=" * 70)
print("TRAINING CLASSIFICATION MODELS")
print("=" * 70)

for model_name, model in models.items():
    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)
    # Train
    model.fit(
        X_train_processed,
        y_train
    )
    # Predictions
    y_pred = model.predict(
        X_test_processed
    )
    # Probability predictions
    if hasattr(model, "predict_proba"):

        y_probability = model.predict_proba(
            X_test_processed
        )[:, 1]
    else:
        y_probability = y_pred

    # METRICS

    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    try:
        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )
    except:
        roc_auc = np.nan

    # DISPLAY RESULTS

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # CONFUSION MATRIX

    cm = confusion_matrix(
        y_test,
        y_pred
    )
    print("\nConfusion Matrix:")
    print(cm)

    # CLASSIFICATION REPORT

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )
    # STORE RESULTS

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc
    })
    trained_models[model_name] = model

# 14. MODEL COMPARISON

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)
print("\n" + "=" * 70)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 70)
print(
    results_df.to_string(
        index=False
    )
)

# 15. SELECT BEST MODEL

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[
    best_model_name
]
print("\n" + "=" * 70)
print("BEST CLASSIFICATION MODEL")
print("=" * 70)
print(
    f"Best Model: {best_model_name}"
)
print(
    f"Best F1 Score: "
    f"{results_df.iloc[0]['F1_Score']:.4f}"
)
print(
    f"Best Accuracy: "
    f"{results_df.iloc[0]['Accuracy']:.4f}"
)
print(
    f"Best Precision: "
    f"{results_df.iloc[0]['Precision']:.4f}"
)
print(
    f"Best Recall: "
    f"{results_df.iloc[0]['Recall']:.4f}"
)
print(
    f"Best ROC-AUC: "
    f"{results_df.iloc[0]['ROC_AUC']:.4f}"
)

# 16. SAVE MODEL COMPARISON

results_path = os.path.join(
    MODEL_DIR,
    "classification_model_comparison.csv"
)
results_df.to_csv(
    results_path,
    index=False
)
print(
    f"\nModel comparison saved to:\n{results_path}"
)

# 17. SAVE BEST MODEL

best_model_path = os.path.join(
    MODEL_DIR,
    "best_classification_model.pkl"
)
joblib.dump(
    best_model,
    best_model_path
)
print(
    f"Best classification model saved to:\n"
    f"{best_model_path}"
)

# 18. SAVE MODEL NAME

model_name_path = os.path.join(
    MODEL_DIR,
    "best_classification_model_name.txt"
)
with open(
    model_name_path,
    "w"
) as file:
    file.write(
        best_model_name
    )

# 19. SAVE FEATURE NAMES

feature_names = preprocessor.get_feature_names_out()
feature_names_path = os.path.join(
    MODEL_DIR,
    "classification_feature_names.txt"
)
with open(
    feature_names_path,
    "w",
    encoding="utf-8"
) as file:
    for feature in feature_names:
        file.write(
            str(feature) + "\n"
        )
print(
    f"\nFeature names saved to:\n"
    f"{feature_names_path}"
)
# 20. FINAL MESSAGE

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL DEVELOPMENT COMPLETED")
print("=" * 70)
print("\nModels trained:")
for model_name in models.keys():
    print(f" - {model_name}")

print("\nTarget:")
print("Good_Investment")

print("\nImportant:")
print("Price_in_Lakhs was NOT used as an ML feature.")

print("\nBest model:")
print(best_model_name)

print("\nFiles created inside models/:")

print("1. classification_preprocessor.pkl")
print("2. best_classification_model.pkl")
print("3. best_classification_model_name.txt")
print("4. classification_model_comparison.csv")
print("5. classification_feature_names.txt")

print("\n" + "=" * 70)