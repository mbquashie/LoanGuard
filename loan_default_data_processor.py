# loan_default_data_processor.py

"""
LoanGuard Data Processor

This script further processes the cleaned Lending Club loan data to prepare it for machine learning models
aimed at predicting loan default risk and determining high credit limits. The processing steps include:
- Splitting the data into training and testing sets with stratification
- Rescaling numerical features using MinMaxScaler
- Balancing the training data using RandomUnderSampler to address class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os

def load_processed_data(file_path):
    """
    Load the processed loan data from a CSV file.

    Parameters:
    - file_path (str): The path to the processed CSV file.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def split_features_target(df, target_column='loan_outcome'):
    """
    Split the DataFrame into features (X) and target (y).

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the target column.

    Returns:
    - X (pd.DataFrame): Features DataFrame.
    - y (pd.Series): Target Series.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def perform_train_test_split(X, y, test_size=0.30, random_state=42):
    """
    Split the data into training and testing sets with stratification.

    Parameters:
    - X (pd.DataFrame): Features DataFrame.
    - y (pd.Series): Target Series.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training target.
    - y_test (pd.Series): Testing target.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def rescale_features(X_train, X_test, feature_columns):
    """
    Rescale numerical features using MinMaxScaler.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - feature_columns (list): List of numerical feature column names to scale.

    Returns:
    - X_train_scaled (pd.DataFrame): Scaled training features.
    - X_test_scaled (pd.DataFrame): Scaled testing features.
    - scaler (MinMaxScaler object): Fitted scaler object.
    """
    scaler = MinMaxScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[feature_columns] = scaler.fit_transform(X_train[feature_columns])
    X_test_scaled[feature_columns] = scaler.transform(X_test[feature_columns])
    
    return X_train_scaled, X_test_scaled, scaler

def balance_training_data(X_train, y_train, random_state=42):
    """
    Balance the training data using RandomUnderSampler to address class imbalance.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_resampled (pd.DataFrame): Resampled training features.
    - y_resampled (pd.Series): Resampled training target.
    - undersampler (RandomUnderSampler object): Fitted undersampler object.
    """
    undersampler = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    
    # Convert to DataFrame and Series for consistency
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)
    
    return X_resampled, y_resampled, undersampler

def save_processed_data(X_train, X_test, y_train, y_test, processed_dir='processed_data'):
    """
    Save the processed training and testing data to CSV files.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training target.
    - y_test (pd.Series): Testing target.
    - processed_dir (str): Directory to save the processed data.
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    print(f"Processed data saved to '{processed_dir}' directory.")

def save_scaler_and_sampler(scaler, undersampler, model_dir='models'):
    """
    Save the scaler and undersampler objects for future use.

    Parameters:
    - scaler (MinMaxScaler object): Fitted scaler.
    - undersampler (RandomUnderSampler object): Fitted undersampler.
    - model_dir (str): Directory to save the scaler and undersampler.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    joblib.dump(scaler, os.path.join(model_dir, 'minmax_scaler.pkl'))
    joblib.dump(undersampler, os.path.join(model_dir, 'random_undersampler.pkl'))
    print(f"Scaler and undersampler saved to '{model_dir}' directory.")

def main():
    # File path to the processed loan data CSV from the data wrangler
    processed_data_input_path = "/Users/mbq/Desktop/Project_Erdos/Data/processed_loan_data.csv"
    
    # Step 1: Load Processed Data
    df = load_processed_data(processed_data_input_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Step 2: Split Features and Target
    X, y = split_features_target(df, target_column='loan_outcome')
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Step 3: Perform Train-Test Split with Stratification
    X_train, X_test, y_train, y_test = perform_train_test_split(X, y, test_size=0.30, random_state=42)
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    print(f"Testing target distribution:\n{y_test.value_counts()}")
    
    # Step 4: Identify Numerical Columns for Rescaling
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"Numerical columns to scale: {numerical_cols}")
    
    # Step 5: Rescale Numerical Features
    X_train_scaled, X_test_scaled, scaler = rescale_features(X_train, X_test, numerical_cols)
    print("Rescaling complete.")
    
    # Step 6: Balance the Training Data using Undersampling
    X_train_balanced, y_train_balanced, undersampler = balance_training_data(X_train_scaled, y_train, random_state=42)
    print(f"Balanced training set shape: {X_train_balanced.shape}, {y_train_balanced.shape}")
    print(f"Balanced training target distribution:\n{y_train_balanced.value_counts()}")
    
    # Step 7: Save the Processed Data
    save_processed_data(X_train_balanced, X_test_scaled, y_train_balanced, y_test, processed_dir='processed_data')
    
    # Step 8: Save the Scaler and Undersampler Objects
    save_scaler_and_sampler(scaler, undersampler, model_dir='models')
    
    print("Data processing complete.")

if __name__ == "__main__":
    main()

