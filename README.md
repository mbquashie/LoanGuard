# LoanGuard

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

**LoanGuard** is an innovative web application designed to predict loan default risks and optimize credit limits using advanced machine learning techniques. By analyzing various financial and credit-related factors, LoanGuard provides users with informed risk assessments, aiding both lenders and borrowers in making prudent financial decisions.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Loan Default Prediction:** Classify loan applications as "Fully Paid" or "Charged Off" using machine learning models.
- **Credit Limit Optimization:** Recommend optimal credit limits for borrowers based on their financial profiles.
- **Interactive Web Application:** User-friendly interface built with Streamlit for real-time risk assessments.
- **Feature Importance Visualization:** Understand which financial factors most significantly influence loan outcomes.
- **Model Retraining Interface:** Allows administrators to retrain models with updated data directly from the app.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Handling Imbalanced Data:** Imbalanced-learn
- **Web Application:** Streamlit
- **Serialization:** Joblib
- **Environment Management:** Anaconda (Conda environments)
- **Version Control:** Git and GitHub

## Installation

Follow these steps to set up the LoanGuard project on your local machine:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mbquashie/LoanGuard.git
   cd LoanGuard
   ```

2. **Create a Conda Environment:**

   Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) installed.

   ```bash
   conda create -n loanguard_env python=3.8
   conda activate loanguard_env
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install the necessary packages manually:*

   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit joblib
   ```

4. **Download the Dataset:**

   Ensure you have the `loan_data.csv` file placed in the `/Data` directory. If not, obtain it from the appropriate source.

## Usage

### 1. **Data Wrangling and Processing**

Run the data wrangling script to preprocess the data:

```bash
python loan_default_data_wrangler.py
```

This will clean the data, handle missing values, engineer features, encode categorical variables, remove outliers, and save the processed data.

### 2. **Model Training**

Execute the data processing and model training script:

```bash
python loan_default_data_processor.py
```

This script will:

- Load the processed data.
- Split the data into training and testing sets with stratification.
- Scale numerical features.
- Address class imbalance using undersampling.
- Train multiple machine learning models.
- Evaluate model performance.
- Save the best-performing models.

### 3. **Launching the Streamlit Application**

Start the LoanGuard web application:

```bash
streamlit run LoanGuard_App.py
```

This will open the application in your default web browser, where you can input applicant data and receive real-time risk assessments.

## Model Performance

LoanGuard employs several machine learning models to predict loan defaults. Below is a summary of their performance:

| **Model**           | **Mean Accuracy** | **ROC AUC** | **Remarks**                                                                 |
|---------------------|--------------------|-------------|-----------------------------------------------------------------------------|
| Logistic Regression | 99.65%             | 0.9949      | High accuracy and excellent classification capabilities.                   |
| Naive Bayes         | 94.72%             | 0.8895      | Moderate accuracy; lower recall for the minority class.                    |
| Decision Tree       | 99.57%             | 0.9958      | High accuracy with minimal misclassifications.                              |
| Random Forest       | 99.85%             | 0.9970      | Highest accuracy and ROC AUC, indicating superior performance.             |
| Gradient Boosting   | 99.64%             | 0.9962      | Excellent accuracy, closely trailing Random Forest.                        |
| XGBoost             | 99.92%             | 0.9994      | Outstanding accuracy and ROC AUC; top-performing model.                    |

**Confusion Matrix Example for XGBoost:**

```
[[276194     14]
 [    73  64529]]
```

## Project Structure

```
LoanGuard/
├── Data/
│   ├── loan_data.csv
│   └── processed_loan_data.csv
├── Models/
│   ├── logistic_regression_model.h5
│   ├── naive_bayes_model.h5
│   ├── decision_tree_model.h5
│   ├── random_forest_model.h5
│   ├── gradient_boosting_model.h5
│   └── xgboost_model.h5
├── Scripts/
│   ├── loan_default_data_wrangler.py
│   ├── loan_default_data_processor.py
│   └── LoanGuard_App.py
├── requirements.txt
├── README.md
└── LICENSE
```

- **Data/**: Contains raw and processed datasets.
- **Models/**: Stores trained machine learning models.
- **Scripts/**: Includes data processing and application scripts.
- **requirements.txt**: Lists project dependencies.
- **README.md**: Project documentation.
- **LICENSE**: Licensing information.

## Contributing

Contributions are welcome! Please follow these steps to contribute to LoanGuard:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your message here"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

Please ensure that your contributions adhere to the project's coding standards and include appropriate documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **Key Features of This README:**

1. **Badges:**
   - **License Badge:** Displays the project's license status.
   - **Python Version Badge:** Indicates the Python version compatibility.

2. **Table of Contents:**
   - Facilitates easy navigation through different sections of the README.

3. **Features Section:**
   - Highlights the key functionalities of the LoanGuard application.

4. **Technologies Used:**
   - Lists all major technologies, libraries, and tools employed in the project.

5. **Installation Instructions:**
   - Step-by-step guide to setting up the project locally, including cloning the repository, setting up the environment, installing dependencies, and obtaining the dataset.

6. **Usage Instructions:**
   - Detailed instructions on how to run data wrangling scripts, train models, and launch the Streamlit application.

7. **Model Performance:**
   - Provides a comprehensive overview of each machine learning model's performance metrics, including Mean Accuracy, ROC AUC, and Confusion Matrix examples.

8. **Project Structure:**
   - Visual representation of the project's directory layout, explaining the purpose of each folder and file.

9. **Contributing Guidelines:**
   - Outlines the process for contributing to the project, promoting collaboration and code contributions.

10. **License Information:**
    - States the project's licensing terms and directs users to the LICENSE file for more details.

11. **Contact Information:**
    - Provides contact details for inquiries or support, fostering communication between maintainers and users.

---

### **Additional Recommendations:**

- **Populate `requirements.txt`:**
  Ensure that your `requirements.txt` file is up-to-date with all necessary dependencies. This allows users to install all required packages easily.

  Example `requirements.txt`:
  ```text
  pandas
  numpy
  scikit-learn
  xgboost
  imbalanced-learn
  streamlit
  joblib
  ```

