# utils/model_loader.py

import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_model(model_path='best_xgboost_model.pkl', encoder_path='label_encoder.pkl'):
    """
    Load the serialized model and label encoder from disk.

    Parameters:
    - model_path (str): Path to the serialized model file.
    - encoder_path (str): Path to the serialized label encoder file.

    Returns:
    - model: Loaded machine learning model.
    - label_encoder: Loaded LabelEncoder.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder file '{encoder_path}' not found.")
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder

def plot_feature_importance(model, feature_names, user_friendly_labels=None, output_path='static/img/feature_importance.png'):
    """
    Plot and save the feature importance.

    Parameters:
    - model: Trained machine learning model.
    - feature_names (list): List of internal feature names.
    - user_friendly_labels (list, optional): List of user-friendly feature labels.
    - output_path (str): Path to save the feature importance plot.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Use user-friendly labels if provided
    if user_friendly_labels:
        labels = [user_friendly_labels[i] for i in indices]
    else:
        labels = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=labels, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
