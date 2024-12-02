# loan_default_data_wrangler.py

"""
LoanGuard Data Wrangler

This script processes the Lending Club loan data to prepare it for machine learning models
aimed at predicting loan default risk and determining high credit limits. The data wrangling
steps include data cleaning, handling missing values, feature engineering, encoding categorical
variables, outlier removal, and encoding the target variable.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load the loan data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    df_main = pd.read_csv(file_path, low_memory=False)
    df = df_main.copy(deep=True)
    print(f"Data loaded with shape: {df.shape}")
    return df

def remove_irrelevant_columns(df):
    """
    Remove columns that are irrelevant or contain unique identifiers.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame after dropping irrelevant columns.
    """
    columns_to_drop = [
        'id', 'member_id', 'emp_title', 'url',
        'title', 'addr_state', 'zip_code'
    ]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    print(f"Removed irrelevant columns. New shape: {df.shape}")
    return df

def filter_target_variable(df):
    """
    Filter the DataFrame to include only 'Fully Paid' and 'Charged Off' loan statuses.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    initial_count = df.shape[0]
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    final_count = df.shape[0]
    print(f"Filtered target variable. Rows before: {initial_count}, after: {final_count}")
    return df

def remove_single_value_columns(df):
    """
    Remove columns that have only a single unique value.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame after dropping single-value columns.
    """
    single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
    if single_value_cols:
        df.drop(columns=single_value_cols, axis=1, inplace=True)
        print(f"Removed single-value columns: {single_value_cols}. New shape: {df.shape}")
    else:
        print("No single-value columns to remove.")
    return df

def handle_missing_values(df):
    """
    Handle missing values by:
    - Dropping columns with >10% missing data.
    - Removing rows with missing values in critical columns.
    - Replacing empty strings with NaN in specific columns.
    - Imputing remaining numerical missing values with the mean.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame after handling missing values.
    """
    # Replace empty strings in 'term' and other relevant columns with NaN
    columns_with_strings = ['term', 'emp_length']  # Add other relevant columns if necessary
    for col in columns_with_strings:
        if col in df.columns:
            df[col].replace('', np.nan, inplace=True)
            print(f"Replaced empty strings with NaN in column: {col}")

    # Calculate percentage of missing values
    missing_percentage = 100 * df.isnull().sum() / len(df)
    
    # Drop columns with >10% missing values
    cols_to_drop = missing_percentage[missing_percentage > 10].index.tolist()
    if cols_to_drop:
        df.drop(columns=cols_to_drop, axis=1, inplace=True)
        print(f"Dropped columns with >10% missing values: {cols_to_drop}. New shape: {df.shape}")
    else:
        print("No columns with >10% missing values to drop.")
    
    # Identify remaining columns with missing values
    remaining_missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"Columns with remaining missing values: {remaining_missing_cols}")
    
    # Define critical columns to drop rows with missing values
    critical_cols = ['emp_length', 'last_credit_pull_d', 'last_pymnt_d']
    missing_in_critical = [col for col in critical_cols if col in remaining_missing_cols]
    if missing_in_critical:
        initial_count = df.shape[0]
        df.dropna(subset=missing_in_critical, how='any', axis=0, inplace=True)
        final_count = df.shape[0]
        print(f"Dropped rows with missing values in critical columns: {missing_in_critical}. Rows before: {initial_count}, after: {final_count}")
    
    # Update remaining missing columns after dropping critical rows
    remaining_missing_cols = [col for col in remaining_missing_cols if col not in critical_cols]
    
    # Impute remaining numerical missing values with mean
    numerical_cols = [col for col in remaining_missing_cols if df[col].dtype in ['float64', 'int64']]
    if numerical_cols:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        print(f"Imputed missing values in numerical columns with mean: {numerical_cols}")
    else:
        print("No numerical columns with missing values to impute.")
    
    # Optionally, handle categorical columns with missing values if any
    categorical_cols = [col for col in remaining_missing_cols if df[col].dtype == 'object']
    if categorical_cols:
        # For simplicity, drop rows with missing categorical values
        initial_count = df.shape[0]
        df.dropna(subset=categorical_cols, how='any', axis=0, inplace=True)
        final_count = df.shape[0]
        print(f"Dropped rows with missing values in categorical columns: {categorical_cols}. Rows before: {initial_count}, after: {final_count}")
    else:
        print("No categorical columns with missing values to handle.")
    
    return df

def convert_and_engineer_features(df):
    """
    Convert and engineer features such as 'term', 'emp_length', and date columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame after feature engineering.
    """
    # Convert 'term' from object to numeric (e.g., '36 months' -> 36)
    if 'term' in df.columns:
        # Handle cases where 'term' might still have empty strings or invalid formats
        df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
        # Check for any remaining NaN in 'term' after extraction
        missing_term = df['term'].isnull().sum()
        if missing_term > 0:
            print(f"Found {missing_term} missing or invalid 'term' entries after extraction. Dropping these rows.")
            df.dropna(subset=['term'], inplace=True)
        df.rename(columns={'term': 'loan_term'}, inplace=True)
        print("Converted 'term' to 'loan_term' as numeric.")
    
    # Process 'emp_length' column
    if 'employment_length' in df.columns:
        # Already converted 'emp_length' to 'employment_length' during missing value handling
        # Here, ensure it's numeric and handle any remaining missing values if necessary
        # If 'employment_length' was filled with 0.5 for '< 1 year', it's already numeric
        pass  # Additional processing can be done here if needed
    elif 'emp_length' in df.columns:
        # In case 'employment_length' wasn't created earlier
        df['employment_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
        df['employment_length'] = df['employment_length'].fillna(0.5)
        df.drop(columns=['emp_length'], axis=1, inplace=True)
        print("Processed 'emp_length' to 'employment_length' as numeric.")
    
    # Split date columns into separate month and year columns
    date_columns = ['earliest_cr_line', 'issue_d', 'last_pymnt_d', 'last_credit_pull_d']
    for col in date_columns:
        if col in df.columns:
            # Extract month and year using regex to handle potential empty strings
            df[f'{col}_month'] = df[col].str.extract(r'(\w+)')[0]
            df[f'{col}_year'] = df[col].str.extract(r'-(\d{4})')[0].astype(float)
            df.drop(columns=[col], axis=1, inplace=True)
            print(f"Split '{col}' into '{col}_month' and '{col}_year' as numeric.")
    
    return df

def encode_target_variable(df):
    """
    Encode the target variable 'loan_status' to binary values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with encoded target variable.
    """
    if 'loan_status' in df.columns:
        df['loan_outcome'] = df['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
        df.drop(columns=['loan_status'], axis=1, inplace=True)
        print("Encoded 'loan_status' to 'loan_outcome' as binary.")
    else:
        print("'loan_status' column not found.")
    return df

def create_dummy_variables(df):
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with dummy variables.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        dummy_df = pd.get_dummies(df[categorical_cols], drop_first=True)
        df = pd.concat([df, dummy_df], axis=1)
        df.drop(columns=categorical_cols, axis=1, inplace=True)
        print(f"Created dummy variables for categorical columns: {categorical_cols}")
    else:
        print("No categorical columns to encode.")
    return df

def remove_outliers(df):
    """
    Remove outliers beyond the 0.5th and 99.9th percentiles for numerical columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame after outlier removal.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    Q1 = df[numeric_cols].quantile(0.005)
    Q3 = df[numeric_cols].quantile(0.999)
    
    # Create a boolean mask for rows within the percentile range
    mask = ~((df[numeric_cols] < Q1) | (df[numeric_cols] > Q3)).any(axis=1)
    initial_count = df.shape[0]
    df_filtered = df[mask]
    final_count = df_filtered.shape[0]
    print(f"Removed outliers. Rows before: {initial_count}, after: {final_count}. Retained: {round(100*(final_count/initial_count),2)}% of data.")
    return df_filtered

def main():
    # File path to the Lending Club loan data CSV
    data_file_path = "/Users/mbq/Desktop/Project_Erdos/Data/loan_data.csv"
    
    # Step 1: Load Data
    df = load_data(data_file_path)
    
    # Step 2: Remove Irrelevant Columns
    df = remove_irrelevant_columns(df)
    
    # Step 3: Filter Target Variable
    df = filter_target_variable(df)
    
    # Step 4: Remove Columns with Single Values
    df = remove_single_value_columns(df)
    
    # Step 5: Handle Missing Values
    df = handle_missing_values(df)
    
    # Step 6: Convert and Engineer Features
    df = convert_and_engineer_features(df)
    
    # Step 7: Encode Target Variable
    df = encode_target_variable(df)
    
    # Step 8: Create Dummy Variables
    df = create_dummy_variables(df)
    
    # Step 9: Remove Outliers
    df = remove_outliers(df)
    
    # Step 10: Save the Processed Data
    processed_data_path = "/Users/mbq/Desktop/Project_Erdos/Data/processed_loan_data.csv"
    df.to_csv(processed_data_path, index=False)
    print(f"Data wrangling complete. Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    main()
