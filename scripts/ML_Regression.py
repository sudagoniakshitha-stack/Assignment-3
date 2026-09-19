
# REGRESSION MODEL DEVELOPMENT
# Target: Future_Price_5Y
# 1. IMPORT LIBRARIES
import os
import warnings
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# 2. IMPORT REGRESSION MODELS

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from scipy.sparse import load_npz

# 3. XGBOOST

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# 4. PROJECT PATHS
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
DATA_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "processed"
)
MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)
os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# 5. DATA FILE PATHS

X_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "X_train_regression.npz"
)
X_TEST_FILE = os.path.join(
    DATA_DIR,
    "X_test_regression.npz"
)
Y_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "y_train_regression.csv"
)
Y_TEST_FILE = os.path.join(
    DATA_DIR,
    "y_test_regression.csv"
)

# 6. LOAD PREPROCESSED REGRESSION DATA

print("=" * 70)
print("REAL ESTATE INVESTMENT ADVISOR")
print("REGRESSION MODEL DEVELOPMENT")
print("=" * 70)
print("\nLoading regression data...")

X_train = load_npz(
    X_TRAIN_FILE
)
X_test = load_npz(
    X_TEST_FILE
)
y_train = pd.read_csv(
    Y_TRAIN_FILE
).squeeze()
y_test = pd.read_csv(
    Y_TEST_FILE
).squeeze()

print("\nRegression data loaded successfully.")
print(
    f"Training shape: {X_train.shape}"
)
print(
    f"Testing shape : {X_test.shape}"
)
print(
    f"Training target rows: {len(y_train)}"
)
print(
    f"Testing target rows : {len(y_test)}"
)

# 7. CHECK TARGET

print("\nRegression target:")
print("Future_Price_5Y")

# 8. DEFINE REGRESSION MODELS

models = {
    "Linear Regression":
        LinearRegression(),
    "Ridge Regression":
        Ridge(
            alpha=1.0
        ),
    "Lasso Regression":
        Lasso(
            alpha=0.001,
            max_iter=2000
        ),
    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=12,
            random_state=42
        ),
}


# 9. ADD XGBOOST
if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
else:
    print("\nWARNING:")
    print("XGBoost is not installed.")
    print(
        "Install it using:"
    )
    print(
        "pip install xgboost"
    )

# 10. TRAINING

results = []
trained_models = {}
print("\n" + "=" * 70)
print("TRAINING REGRESSION MODELS")
print("=" * 70)
for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(
        f"Training: {model_name}"
    )
    print("-" * 70)
    
    # TRAIN MODEL
    
    model.fit(
        X_train,
        y_train
    ) 

    # PREDICTION
   
    y_pred = model.predict(
        X_test
    )
    
    # CALCULATE MSE
    
    mse = mean_squared_error(
        y_test,
        y_pred
    )
    
    # CALCULATE RMSE
    
    rmse = np.sqrt(
        mse
    )

    # CALCULATE MAE
   
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    # CALCULATE R-SQUARED
 
    r2 = r2_score(
        y_test,
        y_pred
    )

    # DISPLAY RESULTS
 
    print(
        f"RMSE: {rmse:.4f}"
    )
    print(
        f"MAE : {mae:.4f}"
    )
    print(
        f"R²  : {r2:.4f}"
    )

    # SAVE RESULTS
  
    results.append({
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2_Score": r2
    })
    trained_models[
        model_name
    ] = model

# 11. MODEL COMPARISON

results_df = pd.DataFrame(
    results
)

# Sort by R² from highest to lowest

results_df = results_df.sort_values(
    by="R2_Score",
    ascending=False
).reset_index(
    drop=True
)
print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)
print(
    results_df.to_string(
        index=False
    )
)

# 12. SELECT BEST MODEL
# Highest R² is considered the best model.
# RMSE and MAE should also be comparatively low.

best_model_name = (
    results_df.iloc[0]["Model"]
)
best_model = trained_models[
    best_model_name
]
best_rmse = (
    results_df.iloc[0]["RMSE"]
)
best_mae = (
    results_df.iloc[0]["MAE"]
)
best_r2 = (
    results_df.iloc[0]["R2_Score"]
)
print("\n" + "=" * 70)
print("BEST REGRESSION MODEL")
print("=" * 70)

print(
    f"Best Model : {best_model_name}"
)
print(
    f"RMSE       : {best_rmse:.4f}"
)
print(
    f"MAE        : {best_mae:.4f}"
)
print(
    f"R²         : {best_r2:.4f}"
)

# 13. SAVE MODEL COMPARISON

comparison_file = os.path.join(
    MODEL_DIR,
    "regression_model_comparison.csv"
)
results_df.to_csv(
    comparison_file,
    index=False
)
print(
    "\nModel comparison saved to:"
)
print(
    comparison_file
)
# 14. SAVE BEST REGRESSION MODEL

best_model_file = os.path.join(
    MODEL_DIR,
    "best_regression_model.pkl"
)

joblib.dump(
    best_model,
    best_model_file
)

print(
    "\nBest regression model saved to:"
)

print(
    best_model_file
)

# 15. SAVE BEST MODEL NAME

best_model_name_file = os.path.join(
    MODEL_DIR,
    "best_regression_model_name.txt"
)
with open(
    best_model_name_file,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        best_model_name
    )

print(
    "\nBest model name saved to:"
)
print(
    best_model_name_file
)
# 16. SAVE FUTURE PRICE FORMULA
# PDF formula:
# Future = Current × (1 + r)^t
# r = 8%
# t = 5 years
# Future_Price_5Y = Current_Price × (1.08)^5

formula_file = os.path.join(
    MODEL_DIR,
    "future_price_formula.txt"
)
with open(
    formula_file,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        "Future Price Formula\n"
    )
    file.write(
        "Future_Price_5Y = Current_Price * (1 + r)^t\n"
    )
    file.write(
        "Annual growth rate (r) = 0.08\n"
    )
    file.write(
        "Time period (t) = 5 years\n"
    )
    file.write(
        "Future_Price_5Y = Current_Price * (1.08)^5\n"
    )
    file.write(
        "Expected appreciation = 46.93%\n"
    )
print(
    "\nFuture price formula saved to:"
)
print(
    formula_file
)
# 17. FINAL SUMMARY

print("\n" + "=" * 70)
print("REGRESSION MODEL DEVELOPMENT COMPLETED")
print("=" * 70)
print("\nModels trained:")
for model_name in models.keys():

    print(
        f" - {model_name}"
    )
print("\nRegression target:")
print("Future_Price_5Y")
print("\nFormula:")
print(
    "Future_Price_5Y = Current_Price * (1.08)^5"
)
print(
    "\nExpected 5-year appreciation: 46.93%"
)
print("\nBest regression model:")
print(
    best_model_name
)
print("\nFiles created inside models/:")

print(
    "1. best_regression_model.pkl"
)
print(
    "2. best_regression_model_name.txt"
)
print(
    "3. regression_model_comparison.csv"
)
print(
    "4. future_price_formula.txt"
)
print("\n" + "=" * 70)