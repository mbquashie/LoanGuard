# utils/data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def prepare_data(file_path):
    """
    Load and preprocess the dataset.

    Parameters:
    - file_path (str): Path to the CSV data file.

    Returns:
    - df_train (pd.DataFrame): Preprocessed training data.
    - df_test (pd.DataFrame): Preprocessed testing data.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Define relevant columns (selected based on VIF analysis)
    selected_features = [
        'avg_cur_bal', 'mo_sin_old_rev_tl_op', 'num_accts_ever_120_pd',
        'num_actv_rev_tl', 'num_bc_tl', 'num_il_tl',
        'num_tl_op_past_12m', 'pub_rec_bankruptcies',
        'tax_liens', 'tot_coll_amt', 'total_il_high_credit_limit'
    ]
    target_variable = 'tot_hi_cred_lim'

    # Filter the dataset
    df = df[selected_features + [target_variable]]

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values. Please handle them before proceeding.")

    # Split into train and test
    df_train, df_test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=100)

    # Rescale the features using MinMaxScaler
    scaler = MinMaxScaler()
    df_train[selected_features] = scaler.fit_transform(df_train[selected_features])
    df_test[selected_features] = scaler.transform(df_test[selected_features])

    return df_train, df_test

