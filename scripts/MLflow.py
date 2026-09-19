
# STEP 6 - MLFLOW EXPERIMENT TRACKING
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

# 1. PROJECT PATHS

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)
MLFLOW_DIR = os.path.join(
    BASE_DIR,
    "mlruns"
)
os.makedirs(
    MLFLOW_DIR,
    exist_ok=True
)

# 2. SET MLFLOW TRACKING LOCATION
mlflow.set_tracking_uri(
    "sqlite:///" + os.path.join(
        BASE_DIR,
        "mlflow.db"
    ).replace("\\", "/")
)
print("\nMLflow tracking database:")
print(
    os.path.join(BASE_DIR, "mlflow.db")
)

# 3. CREATE EXPERIMENTS
CLASSIFICATION_EXPERIMENT = (
    "Real Estate - Classification"
)
REGRESSION_EXPERIMENT = (
    "Real Estate - Regression"
)

# 4. FILE PATHS
classification_results = os.path.join(
    MODEL_DIR,
    "classification_model_comparison.csv"
)
regression_results = os.path.join(
    MODEL_DIR,
    "regression_model_comparison.csv"
)
classification_model = os.path.join(
    MODEL_DIR,
    "best_classification_model.pkl"
)
regression_model = os.path.join(
    MODEL_DIR,
    "best_regression_model.pkl"
)
classification_model_name = os.path.join(
    MODEL_DIR,
    "best_classification_model_name.txt"
)
regression_model_name = os.path.join(
    MODEL_DIR,
    "best_regression_model_name.txt"
)
formula_file = os.path.join(
    MODEL_DIR,
    "future_price_formula.txt"
)
# 5. LOAD MODEL COMPARISON RESULTS
print("=" * 70)
print("REAL ESTATE INVESTMENT ADVISOR")
print("MLFLOW EXPERIMENT TRACKING")
print("=" * 70)
print("\nLoading model comparison results...")
classification_df = pd.read_csv(
    classification_results
)
regression_df = pd.read_csv(
    regression_results
)
print("\nClassification results loaded.")
print(
    classification_df.to_string(
        index=False
    )
)
print("\nRegression results loaded.")
print(
    regression_df.to_string(
        index=False
    )
)
# 6. CLASSIFICATION EXPERIMENT

print("\n" + "=" * 70)
print("MLFLOW - CLASSIFICATION")
print("=" * 70)
mlflow.set_experiment(
    CLASSIFICATION_EXPERIMENT
)
for _, row in classification_df.iterrows():
    model_name = row["Model"]
    with mlflow.start_run(
        run_name=f"Classification - {model_name}"
    ):
        # Log model name
        mlflow.log_param(
            "model",
            model_name
        )
       
        # Log classification metrics
        mlflow.log_metric(
            "accuracy",
            float(row["Accuracy"])
        )
        mlflow.log_metric(
            "precision",
            float(row["Precision"])
        )
        mlflow.log_metric(
            "recall",
            float(row["Recall"])
        )
        mlflow.log_metric(
            "f1_score",
            float(row["F1_Score"])
        )
        mlflow.log_metric(
            "roc_auc",
            float(row["ROC_AUC"])
        )
        print(
            f"Logged classification model: "
            f"{model_name}"
        )

# 7. LOG BEST CLASSIFICATION MODEL
with open(
    classification_model_name,
    "r",
    encoding="utf-8"
) as file:
    best_classification_name = (
        file.read().strip()
    )
best_classification = joblib.load(
    classification_model
)
print(
    "\nBest classification model:",
    best_classification_name
)
with mlflow.start_run(
    run_name="Best Classification Model"
):

    # Log model parameters
    params = (
        best_classification
        .get_params()
    )
    for key, value in params.items():
        try:

            mlflow.log_param(
                key,
                str(value)
            )
        except Exception:
            pass

    # Log best model name

    mlflow.log_param(
        "best_model",
        best_classification_name
    )
   
    # Log best classification metrics
    best_row = classification_df.iloc[0]
    mlflow.log_metric(
        "accuracy",
        float(best_row["Accuracy"])
    )
    mlflow.log_metric(
        "precision",
        float(best_row["Precision"])
    )
    mlflow.log_metric(
        "recall",
        float(best_row["Recall"])
    )
    mlflow.log_metric(
        "f1_score",
        float(best_row["F1_Score"])
    )
    mlflow.log_metric(
        "roc_auc",
        float(best_row["ROC_AUC"])
    )
    # Log model artifact
    
    mlflow.sklearn.log_model(
        best_classification,
        artifact_path="classification_model",
        registered_model_name=(
            "RealEstate_Best_Classification"
        )
    )
    # Log comparison CSV

    mlflow.log_artifact(
        classification_results
    )
print(
    "\nBest classification model logged "
    "and registered."
)

# 8. REGRESSION EXPERIMENT

print("\n" + "=" * 70)
print("MLFLOW - REGRESSION")
print("=" * 70)

mlflow.set_experiment(
    REGRESSION_EXPERIMENT
)
for _, row in regression_df.iterrows():
    model_name = row["Model"]
    with mlflow.start_run(
        run_name=f"Regression - {model_name}"
    ):
        # Log model name
        
        mlflow.log_param(
            "model",
            model_name
        )
        
        # Log regression metrics
       
        mlflow.log_metric(
            "RMSE",
            float(row["RMSE"])
        )
        mlflow.log_metric(
            "MAE",
            float(row["MAE"])
        )
        mlflow.log_metric(
            "R2_Score",
            float(row["R2_Score"])
        )
        print(
            f"Logged regression model: "
            f"{model_name}"
        )

# 9. LOG BEST REGRESSION MODEL
with open(
    regression_model_name,
    "r",
    encoding="utf-8"
) as file:
    best_regression_name = (
        file.read().strip()
    )
best_regression = joblib.load(
    regression_model
)

print(
    "\nBest regression model:",
    best_regression_name
)
with mlflow.start_run(
    run_name="Best Regression Model"
):

    # Log model parameters
    params = (
        best_regression
        .get_params()
    )
    for key, value in params.items():

        try:

            mlflow.log_param(
                key,
                str(value)
            )

        except Exception:
            pass

    # Log best model name    

    mlflow.log_param(
        "best_model",
        best_regression_name
    )

    # Log best regression metrics
    best_row = regression_df.iloc[0]
    mlflow.log_metric(
        "RMSE",
        float(best_row["RMSE"])
    )
    mlflow.log_metric(
        "MAE",
        float(best_row["MAE"])
    )
    mlflow.log_metric(
        "R2_Score",
        float(best_row["R2_Score"])
    )
    # Log regression model artifact
    mlflow.sklearn.log_model(
    best_regression,
    artifact_path="regression_model",
    registered_model_name=(
        "RealEstate_Best_Regression"
    ),
    skops_trusted_types=[
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBRegressor"
    ]
)
    # Log comparison CSV
   
    mlflow.log_artifact(
        regression_results
    )
    # Log future price formula
    
    if os.path.exists(formula_file):
        mlflow.log_artifact(
            formula_file
        )
print(
    "\nBest regression model logged "
    "and registered."
)

# 10. FINAL SUMMARY

print("\n" + "=" * 70)
print("MLFLOW TRACKING COMPLETED SUCCESSFULLY")
print("=" * 70)
print(
    "\nClassification experiment:"
)
print(
    CLASSIFICATION_EXPERIMENT
)
print(
    "\nBest classification model:"
)
print(
    best_classification_name
)
print(
    "\nRegression experiment:"
)
print(
    REGRESSION_EXPERIMENT
)
print(
    "\nBest regression model:"
)
print(
    best_regression_name
)
print(
    "\nMLflow tracking directory:"
)
print(
    MLFLOW_DIR
)
print("\nRegistered models:")
print(
    "1. RealEstate_Best_Classification"
)
print(
    "2. RealEstate_Best_Regression"
)
print("\n" + "=" * 70)