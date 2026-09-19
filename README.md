# Real Estate Investment Advisor

## Project Overview

The **Real Estate Investment Advisor** is a machine learning-based application that analyzes residential property data to help users evaluate property investment potential and estimate future property value.

The project performs data cleaning, exploratory data analysis, feature engineering, machine learning classification and regression, MLflow experiment tracking, and provides an interactive Streamlit application for property investment analysis.

## Features

- Property Data Cleaning
- Exploratory Data Analysis
- Price and Property Size Analysis
- Price per Square Foot Analysis
- City, State and Locality Analysis
- Feature Engineering
- Good Investment Classification
- Future Price Prediction
- Multiple Classification Models
- Multiple Regression Models
- Model Performance Evaluation
- MLflow Experiment Tracking
- Interactive Streamlit Application
- Investment Prediction
- Five-Year Future Price Estimation
- Model Confidence
- Feature Importance
- Interactive Visualizations


## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- MLflow
- Streamlit
- Joblib

## Machine Learning

### Classification

**Target:** `Good_Investment`

Models used:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

### Regression

**Target:** `Future_Price_5Y`

Models used:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- XGBoost Regressor

## Model Evaluation

### Classification

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Regression

- RMSE
- MAE
- R² Score

## MLflow

MLflow is used for machine learning experiment tracking and model management.

It tracks:

- Model parameters
- Evaluation metrics
- Model artifacts
- Classification experiments
- Regression experiments
- Best-performing models
- Model registration

## Streamlit Application

The Streamlit application provides:

- Property information input
- Good Investment prediction
- Classification confidence
- Estimated Price after 5 Years
- Feature importance
- Property insights
- Interactive visualizations

## Project Structure

```text
REAL_ESTATE_INVESTMENT_ADVISOR/
│
├── app.py
│
├── scripts/
│   ├── clean dataset.py
│   ├── EDA.py
│   ├── ML_data.py
│   ├── ML_preprocessing.py
│   ├── ML_classification.py
│   ├── ML_Regression.py
│   └── MLflow.py
│
└── README.md
