## app.py
#
#import streamlit as st
#import pandas as pd
#import numpy as np
#import os
#from utils.model_loader import load_model, plot_feature_importance
#import matplotlib.pyplot as plt
#import seaborn as sns
#import joblib
#import warnings
#from utils.data_preparation import prepare_data
#from utils.feature_mapping import FEATURE_MAPPING  # Import the mapping
#
## Suppress warnings
#warnings.filterwarnings('ignore')
#
## Set Streamlit page configuration with a custom layout and theme
#st.set_page_config(
#    page_title="Loan Guard",
#    page_icon="🔒💳",
#    layout="wide",
#    initial_sidebar_state="expanded",
#)
#
## Apply custom CSS for styling
#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
#
## Assuming you have a 'style.css' file in the 'static/css/' directory
#if os.path.exists('static/css/style.css'):
#    local_css('static/css/style.css')
#
## Title with an icon
#st.markdown("""
#    <h1 style='text-align: center; color: #0E4B5B;'>
#        <img src='https://img.icons8.com/fluency/48/000000/loan.png' style='vertical-align: middle;'/> 
#        Loan Guard
#    </h1>
#    """, unsafe_allow_html=True)
#
## Add a brief description
#st.markdown("""
#    <p style='text-align: center; color: #555555;'>
#        Assess the risk of loan default with Loan Guard. Enter your financial details below to check your risk status.
#    </p>
#    """, unsafe_allow_html=True)
#
## Sidebar for user inputs with a background color
#st.sidebar.header("🔍 Enter Your Financial Details")
#
## Define the features used in the model
#SELECTED_FEATURES = list(FEATURE_MAPPING.keys())
#
## Function to get user input with user-friendly labels
#def get_user_input():
#    user_data = {}
#    for feature, label in FEATURE_MAPPING.items():
#        if 'num_' in feature or 'tax_liens' in feature or 'pub_rec_bankruptcies' in feature:
#            step = 1  # Integer inputs
#            min_val = 0
#            value = 0
#        else:
#            step = 0.1  # Float inputs
#            min_val = 0.0
#            value = 0.0
#        user_data[feature] = st.sidebar.number_input(
#            label=label,  # Display user-friendly label
#            value=value,
#            min_value=min_val,
#            step=step,
#            help=f"Enter the {label.lower()}."
#        )
#    return user_data
#
## Collect user input
#user_input = get_user_input()
#
## Validate and convert inputs
#def validate_and_convert(inputs):
#    validated_data = {}
#    for feature, value in inputs.items():
#        if value < 0:
#            st.error(f"Invalid input for '{FEATURE_MAPPING[feature]}'. Please enter a non-negative value.")
#            return None
#        # Example: Ensure 'Months Since Oldest Revolving Trade Line Open' is within a realistic range
#        if feature == 'mo_sin_old_rev_tl_op' and value > 600:
#            st.error(f"'{FEATURE_MAPPING[feature]}' seems unusually high. Please verify the input.")
#            return None
#        validated_data[feature] = float(value)
#    return validated_data
#
#validated_input = validate_and_convert(user_input)
#
## Load the trained model
#MODEL_PATH = 'best_xgboost_model.pkl'
#if os.path.exists(MODEL_PATH):
#    model = load_model(MODEL_PATH)
#else:
#    model = None
#    st.warning("Trained model not found. Please train the model first.")
#
## Feature Importance Section
#if model:
#    st.subheader("📊 Feature Importance")
#    feature_imp_fig, ax = plt.subplots(figsize=(10,6))
#    importances = model.feature_importances_
#    indices = np.argsort(importances)[::-1]
#    sorted_importances = importances[indices]
#    sorted_labels = [FEATURE_MAPPING[feat] for feat in SELECTED_FEATURES]
#    
#    sns.barplot(x=sorted_importances, y=sorted_labels, palette='viridis', ax=ax)
#    plt.title('Feature Importance', fontsize=16)
#    plt.xlabel('Importance', fontsize=14)
#    plt.ylabel('Feature', fontsize=14)
#    plt.tight_layout()
#    st.pyplot(feature_imp_fig)
#
## Prediction Button
#if st.button("Check Risk"):
#    if validated_input and model:
#        input_df = pd.DataFrame([validated_input])
#        prediction = model.predict(input_df)[0]
#        
#        # Map numerical prediction to risk categories
#        risk_mapping = {
#            'High Risk': 'High Risk of Default',
#            'Moderate Risk': 'Moderate Risk of Default',
#            'Low Risk': 'Low Risk of Default'
#        }
#        
#        risk_status = risk_mapping.get(prediction, "Unknown Risk")
#        
#        # Define colors and icons based on risk category
#        risk_colors = {
#            'High Risk of Default': "#FF4B4B",  # Red
#            'Moderate Risk of Default': "#FFA500",  # Orange
#            'Low Risk of Default': "#4CAF50"  # Green
#        }
#        
#        risk_icons = {
#            'High Risk of Default': "⚠️",
#            'Moderate Risk of Default': "⚠️",
#            'Low Risk of Default': "✅"
#        }
#        
#        color = risk_colors.get(risk_status, "#555555")
#        icon = risk_icons.get(risk_status, "")
#        
#        st.markdown(f"""
#            <div style='background-color: {color}; padding: 20px; border-radius: 10px;'>
#                <h2 style='color: white; text-align: center;'>{icon} {risk_status}</h2>
#            </div>
#        """, unsafe_allow_html=True)
#        
#        st.markdown(f"""
#            <p style='text-align: center; color: #555555;'>
#                Based on your financial details, your assessed risk to default on a loan is <strong>{risk_status}</strong>.
#            </p>
#        """, unsafe_allow_html=True)
#    elif not model:
#        st.error("Trained model is not available. Please train the model first.")
#
## Model Training Section
#st.sidebar.header("🛠️ Train XGBoost Model")
#if st.sidebar.button("Start Training"):
#    with st.spinner('Loading and preparing data...'):
#        try:
#            df_train, df_test = prepare_data('data/processed_loan_data.csv')
#            st.success('Data preparation complete.')
#        except Exception as e:
#            st.error(f"Error in data preparation: {e}")
#            st.stop()
#
#    # Define features and target
#    X_train = df_train[SELECTED_FEATURES]
#    y_train = df_train['risk_category']  # Updated target for classification
#    X_test = df_test[SELECTED_FEATURES]
#    y_test = df_test['risk_category']
#
#    # Define the model and hyperparameter grid
#    xgb = XGBClassifier(objective='multi:softprob', num_class=3, random_state=100, use_label_encoder=False, eval_metric='mlogloss')
#
#    param_grid = {
#        'n_estimators': [100, 200, 300],
#        'max_depth': [3, 5, 7],
#        'learning_rate': [0.01, 0.1, 0.2],
#        'subsample': [0.8, 1.0],
#        'colsample_bytree': [0.8, 1.0]
#    }
#
#    # Initialize GridSearchCV with verbose output
#    grid_search = GridSearchCV(
#        estimator=xgb,
#        param_grid=param_grid,
#        scoring='accuracy',
#        cv=3,
#        verbose=3,  # Adjust verbosity as needed
#        n_jobs=-1
#    )
#
#    # Fit GridSearchCV
#    with st.spinner('Starting hyperparameter tuning with GridSearchCV...'):
#        try:
#            grid_search.fit(X_train, y_train)
#            st.success('Hyperparameter tuning complete.')
#        except Exception as e:
#            st.error(f"Error during hyperparameter tuning: {e}")
#            st.stop()
#
#    # Evaluate the best model
#    best_xgb = grid_search.best_estimator_
#    best_params = grid_search.best_params_
#    y_pred = best_xgb.predict(X_test)
#    accuracy = accuracy_score(y_test, y_pred)
#    report = classification_report(y_test, y_pred)
#
#    st.write(f"**Best Parameters:** {best_params}")
#    st.write(f"**Test Accuracy:** {accuracy:.4f}")
#    st.text("**Classification Report:**")
#    st.text(report)
#
#    # Save the best model
#    joblib.dump(best_xgb, MODEL_PATH)
#    st.success(f"Best XGBoost model saved as '{MODEL_PATH}'.")
#
#    # Update the feature importance plot
#    user_friendly_labels = [FEATURE_MAPPING[feat] for feat in SELECTED_FEATURES]
#    plot_feature_importance(best_xgb, SELECTED_FEATURES, user_friendly_labels, FEATURE_IMPORTANCE_PATH)
#    st.experimental_rerun()
#
## Display a download link for the trained model
#if os.path.exists(MODEL_PATH):
#    with open(MODEL_PATH, 'rb') as f:
#        st.sidebar.download_button(
#            label="📥 Download Trained Model",
#            data=f,
#            file_name='best_xgboost_model.pkl',
#            mime='application/octet-stream'
#        )

#
#import streamlit as st
#import pandas as pd
#import numpy as np
#import os
#from utils.model_loader import load_model, plot_feature_importance
#import matplotlib.pyplot as plt
#import seaborn as sns
#import joblib
#import warnings
#from utils.data_preparation import prepare_data
#from utils.feature_mapping import FEATURE_MAPPING  # Import the mapping
#
## Suppress warnings
#warnings.filterwarnings('ignore')
#
## Set Streamlit page configuration with a custom layout and theme
#st.set_page_config(
#    page_title="Loan Guard",
#    page_icon="🔒💳",
#    layout="wide",
#    initial_sidebar_state="expanded",
#)
#
## Apply custom CSS for styling
#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
#
## Assuming you have a 'style.css' file in the 'static/css/' directory
#if os.path.exists('static/css/style.css'):
#    local_css('static/css/style.css')
#
## Title with an icon
#st.markdown("""
#    <h1 style='text-align: center; color: #0E4B5B;'>
#        <img src='https://img.icons8.com/fluency/48/000000/loan.png' style='vertical-align: middle;'/> 
#        Loan Guard
#    </h1>
#    """, unsafe_allow_html=True)
#
## Add a brief description
#st.markdown("""
#    <p style='text-align: center; color: #555555;'>
#        Assess the risk of loan default with Loan Guard. Enter your financial details below to check your risk status.
#    </p>
#    """, unsafe_allow_html=True)
#
## Sidebar for user inputs with a background color
#st.sidebar.header("🔍 Enter Your Financial Details")
#
## Define the features used in the model
#SELECTED_FEATURES = list(FEATURE_MAPPING.keys())
#
## Function to get user input with user-friendly labels
#def get_user_input():
#    user_data = {}
#    for feature, label in FEATURE_MAPPING.items():
#        if 'num_' in feature or 'tax_liens' in feature or 'pub_rec_bankruptcies' in feature:
#            step = 1  # Integer inputs
#            min_val = 0
#            value = 0
#        else:
#            step = 0.1  # Float inputs
#            min_val = 0.0
#            value = 0.0
#        user_data[feature] = st.sidebar.number_input(
#            label=label,  # Display user-friendly label
#            value=value,
#            min_value=min_val,
#            step=step,
#            help=f"Enter the {label.lower()}."
#        )
#    return user_data
#
## Collect user input
#user_input = get_user_input()
#
## Validate and convert inputs
#def validate_and_convert(inputs):
#    validated_data = {}
#    for feature, value in inputs.items():
#        if value < 0:
#            st.error(f"Invalid input for '{FEATURE_MAPPING[feature]}'. Please enter a non-negative value.")
#            return None
#        # Example: Ensure 'Months Since Oldest Revolving Trade Line Open' is within a realistic range
#        if feature == 'mo_sin_old_rev_tl_op' and value > 600:
#            st.error(f"'{FEATURE_MAPPING[feature]}' seems unusually high. Please verify the input.")
#            return None
#        validated_data[feature] = float(value)
#    return validated_data
#
#validated_input = validate_and_convert(user_input)
#
## Load the trained model
#MODEL_PATH = 'best_xgboost_model.pkl'
#if os.path.exists(MODEL_PATH):
#    model = load_model(MODEL_PATH)
#else:
#    model = None
#    st.warning("Trained model not found. Please train the model first.")
#
## Feature Importance Section
#if model:
#    st.subheader("📊 Feature Importance")
#    feature_imp_fig, ax = plt.subplots(figsize=(10,6))
#    importances = model.feature_importances_
#    indices = np.argsort(importances)[::-1]
#    sorted_importances = importances[indices]
#    sorted_labels = [FEATURE_MAPPING[feat] for feat in SELECTED_FEATURES]
#    
#    sns.barplot(x=sorted_importances, y=sorted_labels, palette='viridis', ax=ax)
#    plt.title('Feature Importance', fontsize=16)
#    plt.xlabel('Importance', fontsize=14)
#    plt.ylabel('Feature', fontsize=14)
#    plt.tight_layout()
#    st.pyplot(feature_imp_fig)
#
## Prediction Button
#if st.button("Check Risk"):
#    if validated_input and model:
#        input_df = pd.DataFrame([validated_input])
#        prediction = model.predict(input_df)[0]
#        prediction = round(prediction, 2)
#        
#        # Define a threshold for risk classification
#        # This threshold should be determined based on your business logic or data analysis
#        # For demonstration, let's assume:
#        # If predicted credit limit is below $10,000 -> High Risk
#        # Else -> Low Risk
#        threshold = 10000  # Example threshold
#        
#        if prediction < threshold:
#            risk_status = "High Risk of Default"
#            risk_color = "#FF4B4B"  # Red color
#            risk_icon = "⚠️"
#        else:
#            risk_status = "Low Risk of Default"
#            risk_color = "#4CAF50"  # Green color
#            risk_icon = "✅"
#        
#        st.markdown(f"""
#            <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px;'>
#                <h2 style='color: white; text-align: center;'>{risk_icon} {risk_status}</h2>
#            </div>
#        """, unsafe_allow_html=True)
#        
#        st.markdown(f"""
#            <p style='text-align: center; color: #555555;'>
#                Based on your financial details, your risk to default on a loan is assessed as <strong>{risk_status}</strong>.
#            </p>
#        """, unsafe_allow_html=True)
#    elif not model:
#        st.error("Trained model is not available. Please train the model first.")
#
## Model Training Section
#st.sidebar.header("🛠️ Train XGBoost Model")
#if st.sidebar.button("Start Training"):
#    with st.spinner('Loading and preparing data...'):
#        try:
#            df_train, df_test = prepare_data('data/processed_loan_data.csv')
#            st.success('Data preparation complete.')
#        except Exception as e:
#            st.error(f"Error in data preparation: {e}")
#            st.stop()
#
#    # Define features and target
#    X_train = df_train[SELECTED_FEATURES]
#    y_train = df_train['tot_hi_cred_lim']
#    X_test = df_test[SELECTED_FEATURES]
#    y_test = df_test['tot_hi_cred_lim']
#
#    # Define the model and hyperparameter grid
#    xgb = XGBRegressor(objective='reg:squarederror', random_state=100)
#
#    param_grid = {
#        'n_estimators': [100, 200, 300],
#        'max_depth': [3, 5, 7],
#        'learning_rate': [0.01, 0.1, 0.2],
#        'subsample': [0.8, 1.0],
#        'colsample_bytree': [0.8, 1.0]
#    }
#
#    # Initialize GridSearchCV with verbose output
#    grid_search = GridSearchCV(
#        estimator=xgb,
#        param_grid=param_grid,
#        scoring='r2',
#        cv=3,
#        verbose=3,  # Adjust verbosity as needed
#        n_jobs=-1
#    )
#
#    # Fit GridSearchCV
#    with st.spinner('Starting hyperparameter tuning with GridSearchCV...'):
#        try:
#            grid_search.fit(X_train, y_train)
#            st.success('Hyperparameter tuning complete.')
#        except Exception as e:
#            st.error(f"Error during hyperparameter tuning: {e}")
#            st.stop()
#
#    # Evaluate the best model
#    best_xgb = grid_search.best_estimator_
#    best_params = grid_search.best_params_
#    train_r2 = r2_score(y_train, best_xgb.predict(X_train))
#    test_r2 = r2_score(y_test, best_xgb.predict(X_test))
#
#    st.write(f"**Best Parameters:** {best_params}")
#    st.write(f"**Train R²:** {train_r2:.4f}")
#    st.write(f"**Test R²:** {test_r2:.4f}")
#
#    # Save the best model
#    joblib.dump(best_xgb, MODEL_PATH)
#    st.success(f"Best XGBoost model saved as '{MODEL_PATH}'.")
#
#    # Update the feature importance plot
#    user_friendly_labels = [FEATURE_MAPPING[feat] for feat in SELECTED_FEATURES]
#    plot_feature_importance(best_xgb, SELECTED_FEATURES, user_friendly_labels, FEATURE_IMPORTANCE_PATH)
#    st.experimental_rerun()
#
## Display a download link for the trained model
#if os.path.exists(MODEL_PATH):
#    with open(MODEL_PATH, 'rb') as f:
#        st.sidebar.download_button(
#            label="📥 Download Trained Model",
#            data=f,
#            file_name='best_xgboost_model.pkl',
#            mime='application/octet-stream'
#        )




# app.py
# app.py
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.model_loader import load_model, plot_feature_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from utils.data_preparation import prepare_data
from utils.feature_mapping import FEATURE_MAPPING  # Import the mapping
import plotly.express as px

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Streamlit page configuration with a custom layout and theme
st.set_page_config(
    page_title="Loan Guard",
    page_icon="🔒💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for styling
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{file_name}' not found. Proceeding without custom styles.")

# Assuming you have a 'style.css' file in the 'static/css/' directory
local_css('static/css/style.css')

# Title with an icon
st.markdown("""
    <h1 style='text-align: center; color: #0E4B5B;'>
        <img src='https://icons8.com/icon/Co3eCE9WwGnj/loan' style='vertical-align: middle;'/> 
        Loan Guard
    </h1>
    """, unsafe_allow_html=True)

# Add a brief description
st.markdown("""
    <p style='text-align: center; color: #555555;'>
        Assess the risk of loan default with Loan Guard. Enter your financial details below to check your risk status.
    </p>
    """, unsafe_allow_html=True)

# Sidebar for user inputs with a background color
st.sidebar.header("🔍 Enter Your Financial Details")

# Define the features used in the model
SELECTED_FEATURES = list(FEATURE_MAPPING.keys())

# Function to get user input with user-friendly labels
def get_user_input():
    user_data = {}
    for feature, label in FEATURE_MAPPING.items():
        if 'num_' in feature or 'tax_liens' in feature or 'pub_rec_bankruptcies' in feature:
            step = 1  # Integer inputs
            min_val = 0
            value = 0
        else:
            step = 0.1  # Float inputs
            min_val = 0.0
            value = 0.0
        user_data[feature] = st.sidebar.number_input(
            label=label,  # Display user-friendly label
            value=value,
            min_value=min_val,
            step=step,
            help=f"Enter the {label.lower()}."
        )
    return user_data

# Collect user input
user_input = get_user_input()

# Validate and convert inputs
def validate_and_convert(inputs):
    validated_data = {}
    for feature, value in inputs.items():
        if value < 0:
            st.error(f"Invalid input for '{FEATURE_MAPPING[feature]}'. Please enter a non-negative value.")
            return None
        # Example: Ensure 'Months Since Oldest Revolving Trade Line Open' is within a realistic range
        if feature == 'mo_sin_old_rev_tl_op' and value > 600:
            st.error(f"'{FEATURE_MAPPING[feature]}' seems unusually high. Please verify the input.")
            return None
        validated_data[feature] = float(value)
    return validated_data

validated_input = validate_and_convert(user_input)

# Load the trained model and label encoder
@st.cache_resource
def load_trained_model(model_path, encoder_path):
    return load_model(model_path, encoder_path)

MODEL_PATH = 'best_xgboost_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model, label_encoder = load_trained_model(MODEL_PATH, ENCODER_PATH)
else:
    model = None
    label_encoder = None
    st.warning("Trained model or label encoder not found. Please train the model first.")

# Feature Importance Section
if model:
    st.subheader("📊 Feature Importance")
    
    # Generate feature importance data
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_labels = [FEATURE_MAPPING[feat] for feat in SELECTED_FEATURES]
    
    # Create a DataFrame for Plotly
    df_importance = pd.DataFrame({
        'Feature': [sorted_labels[i] for i in range(len(sorted_labels))],
        'Importance': sorted_importances
    })
    
    # Plot using Plotly for interactivity
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# Prediction Button
if st.button("Check Risk"):
    if validated_input and model and label_encoder:
        input_df = pd.DataFrame([validated_input])
        prediction_encoded = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        st.write(f"**Raw Prediction:** {prediction}")  # Debugging line
        
        # Define risk mapping
        risk_mapping = {
            'high': 'High Risk of Default',
            'moderate': 'Moderate Risk of Default',
            'low': 'Low Risk of Default'
        }
        
        risk_status = risk_mapping.get(prediction.lower(), "Unknown Risk")
        
        if risk_status != "Unknown Risk":
            # Define colors and icons based on risk category
            risk_colors = {
                'High Risk of Default': "#FF4B4B",  # Red
                'Moderate Risk of Default': "#FFA500",  # Orange
                'Low Risk of Default': "#4CAF50"  # Green
            }
            
            risk_icons = {
                'High Risk of Default': "⚠️",
                'Moderate Risk of Default': "⚠️",
                'Low Risk of Default': "✅"
            }
            
            color = risk_colors.get(risk_status, "#555555")
            icon = risk_icons.get(risk_status, "")
            
            st.markdown(f"""
                <div style='background-color: {color}; padding: 20px; border-radius: 10px;'>
                    <h2 style='color: white; text-align: center;'>{icon} {risk_status}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p style='text-align: center; color: #555555;'>
                    Based on your financial details, your assessed risk to default on a loan is <strong>{risk_status}</strong>.
                </p>
            """, unsafe_allow_html=True)
        else:
            st.error("Unknown Risk. Please contact support.")
    elif not model:
        st.error("Trained model is not available. Please train the model first.")

# Model Training Section
st.sidebar.header("🛠️ Train XGBoost Model")
if st.sidebar.button("Start Training"):
    with st.spinner('Loading and preparing data...'):
        try:
            df_train, df_test = prepare_data('data/processed_loan_data.csv')
            st.success('Data preparation complete.')
        except Exception as e:
            st.error(f"Error in data preparation: {e}")
            st.stop()

    # Define features and target
    X_train = df_train[SELECTED_FEATURES]
    y_train = df_train['risk_category']  # Updated target for classification
    X_test = df_test[SELECTED_FEATURES]
    y_test = df_test['risk_category']

    # Define the model and hyperparameter grid
    xgb = XGBClassifier(objective='multi:softprob', num_class=3, random_state=100, eval_metric='mlogloss')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize GridSearchCV with verbose output
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=3,  # Adjust verbosity as needed
        n_jobs=-1
    )

    # Fit GridSearchCV
    with st.spinner('Starting hyperparameter tuning with GridSearchCV...'):
        try:
            grid_search.fit(X_train, y_train)
            st.success('Hyperparameter tuning complete.')
        except Exception as e:
            st.error(f"Error during hyperparameter tuning: {e}")
            st.stop()

    # Evaluate the best model
    best_xgb = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=best_xgb.classes_)

    st.write(f"**Best Parameters:** {best_params}")
    st.write(f"**Test Accuracy:** {accuracy:.4f}")
    st.text("**Classification Report:**")
    st.text(report)

    # Save the best model and label encoder
    joblib.dump(best_xgb, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)  # Ensure label encoder is saved during training
    st.success(f"Best XGBoost model saved as '{MODEL_PATH}'.")
    st.success(f"Label Encoder saved as '{ENCODER_PATH}'.")

    # Update the feature importance plot
    user_friendly_labels = [FEATURE_MAPPING[feat] for feat in SELECTED_FEATURES]
    plot_feature_importance(best_xgb, SELECTED_FEATURES, user_friendly_labels, 'static/img/feature_importance.png')
    st.experimental_rerun()

# Display a download link for the trained model
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    with open(MODEL_PATH, 'rb') as f_model, open(ENCODER_PATH, 'rb') as f_encoder:
        st.sidebar.download_button(
            label="📥 Download Trained Model",
            data=f_model,
            file_name='best_xgboost_model.pkl',
            mime='application/octet-stream'
        )
        st.sidebar.download_button(
            label="📥 Download Label Encoder",
            data=f_encoder,
            file_name='label_encoder.pkl',
            mime='application/octet-stream'
        )
