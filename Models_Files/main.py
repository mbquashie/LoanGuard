import pandas as pd
from logistic_regression import train_logistic_regression
from naive_bayes import train_naive_bayes
from decision_tree import train_decision_tree
from random_forest import train_random_forest
from gradient_boosting import train_gradient_boosting
from xgboost_model import train_xgboost

# Load the datasets
print("Loading data...")
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()  # Convert to Series
y_test = pd.read_csv("y_test.csv").squeeze()    # Convert to Series

# Ensure correct loading
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Run each model
train_logistic_regression(X_train, y_train, X_test, y_test)
train_naive_bayes(X_train, y_train, X_test, y_test)
train_decision_tree(X_train, y_train, X_test, y_test)
train_random_forest(X_train, y_train, X_test, y_test)
train_gradient_boosting(X_train, y_train, X_test, y_test)
train_xgboost(X_train, y_train, X_test, y_test)

print("All models trained and saved successfully.")

