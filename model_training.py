# model_training.py

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import warnings
from utils.data_preparation import prepare_data

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def train_xgboost_model(file_path, model_output_path='best_xgboost_model.pkl', encoder_output_path='label_encoder.pkl'):
    """
    Train an XGBoost classification model with hyperparameter tuning.

    Parameters:
    - file_path (str): Path to the CSV data file.
    - model_output_path (str): Path to save the trained model.
    - encoder_output_path (str): Path to save the label encoder.

    Returns:
    - best_xgb (XGBClassifier): The best XGBoost model after hyperparameter tuning.
    - accuracy (float): Accuracy score on the test data.
    - classification_report (str): Detailed classification report.
    - best_params (dict): Best hyperparameters found during tuning.
    """
    # Step 1: Prepare the data
    print("Loading and preparing data...")
    df_train, df_test = prepare_data(file_path)
    print("Data preparation complete.\n")

    # Define features and target
    selected_features = [
        'avg_cur_bal', 'mo_sin_old_rev_tl_op', 'num_accts_ever_120_pd',
        'num_actv_rev_tl', 'num_bc_tl', 'num_il_tl',
        'num_tl_op_past_12m', 'pub_rec_bankruptcies',
        'tax_liens', 'tot_coll_amt', 'total_il_high_credit_limit'
    ]
    target = 'risk_category'  # New target for classification

    # Create risk categories based on 'tot_hi_cred_lim'
    def categorize_risk(x):
        if x < 10000:
            return 'high'  # Changed to lowercase
        elif 10000 <= x < 20000:
            return 'moderate'
        else:
            return 'low'

    df_train['risk_category'] = df_train['tot_hi_cred_lim'].apply(categorize_risk)
    df_test['risk_category'] = df_test['tot_hi_cred_lim'].apply(categorize_risk)

    X_train = df_train[selected_features]
    y_train = df_train[target]
    X_test = df_test[selected_features]
    y_test = df_test[target]

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Step 2: Define the model and hyperparameter grid
    xgb = XGBClassifier(objective='multi:softprob', num_class=3, random_state=100, eval_metric='mlogloss')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Step 3: Initialize GridSearchCV with verbose output
    print("Starting hyperparameter tuning with GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=3,  # Adjust verbosity as needed
        n_jobs=-1
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train_encoded)
    print("Hyperparameter tuning complete.\n")

    # Step 4: Evaluate the best model
    best_xgb = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred_encoded = best_xgb.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    report = classification_report(y_test_encoded, y_pred_encoded, target_names=label_encoder.classes_)

    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Step 5: Save the best model and label encoder
    joblib.dump(best_xgb, model_output_path)
    joblib.dump(label_encoder, encoder_output_path)
    print(f"Best XGBoost model saved as '{model_output_path}'.")
    print(f"Label Encoder saved as '{encoder_output_path}'.\n")

    return best_xgb, accuracy, report, best_params

if __name__ == "__main__":
    # Example usage
    train_xgboost_model(file_path='data/processed_loan_data.csv')
