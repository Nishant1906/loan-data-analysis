#  Loan Default Prediction Pipeline

This repository builds a full machine learning pipeline to predict loan default risk. It covers data cleaning, feature engineering, exploratory data analysis (EDA), model experimentation, hyperparameter tuning, and final model evaluation.

---

## 📁 Project Structure

```text
loan-data-analysis/
│
├── data-wrangling.ipynb       # Data cleaning, feature engineering, EDA
├── model-training.ipynb       # Model experiments, tuning, final evaluation
├── EDA_html_reports/          # Full EDA reports (HTML)
├── processed_data/            # Cleaned & transformed datasets
├── models/                    # Saved models
├── flask-deployment/          # Files to expose ML model as Flask REST API
├── README.md                  # You are here
└── ...


---

##  1. Data Wrangling and EDA (`data-wrangling.ipynb`)

This notebook handles all data preprocessing tasks:

###  Steps Performed:

- **Data Cleaning**
  - Fixed inconsistent datatypes (e.g., dates, categoricals)
  - Handled duplicate rows if any

- **Outlier Treatment**
  - Clipped extreme values to reduce skew and influence on models

- **Feature Engineering**
  - Created domain-relevant features like:
    - `Income_to_Loan_Ratio`
    - `Installment_Rate`
    - `Employment_Percent_Life`
    - `Income_to_Region_Pop`
    - `Occupation_Risk_Level`

- **Missing Values Handling**
  - Dropped columns with **too many missing values**
  - Imputed remaining missing values using suitable strategies (median, mode, etc.)

- **Encoding**
  - Applied:
    - **Label Encoding** to ordinal variables
    - **One-Hot Encoding** to nominal categorical variables

- **Exploratory Data Analysis (EDA)**
  - Visualized key relationships, distributions, class imbalance
  - Analyzed default patterns across features

###  EDA Reports:
- Full automated profiling reports generated using **YData Profiling** (formerly pandas-profiling)
- Located under `EDA_html_reports/`

>  These `.html` files are heavy and **may not render properly on GitHub**.  
> Please **download and open them locally in a browser** for full interactivity.

---

##  2. Model Training & Evaluation (`model-training.ipynb`)

###  Workflow Followed:

1. **Data Import**
   - Loaded cleaned dataset from previous step

2. **Model Experiments**
   - Tested various models:
     - Logistic Regression
     - Random Forest
     - XGBoost
     - BalancedRandomForest (imblearn)
   - Used different resampling techniques:
     - **SMOTE**
     - **Random Over/Under-sampling**
     - **BalancedRandomForest** (built-in undersampling)
   - Compared using multiple thresholds (e.g., 0.3, 0.5, 0.8) for imbalanced classification

3. **Model Selection**
   - Selected best model based on **Recall** and **Precision**
   - Balanced importance: prioritize Recall, but maintain acceptable Precision

4. **Hyperparameter Tuning**
   - Used **Optuna** for tuning with custom **composite scoring formula**:
     ```
     composite_score = 0.4 * recall + 0.4 * f1 + 0.1 * pr_auc
     ```
   - Tracked all metrics using **MLflow**

5. **Final Model**
   - Trained final model on best config
   - Evaluated on test set
   - Saved model using `joblib` as `.pkl` file

6. **Feature Importance**
   - Plotted top important features based on the selected model

---


##  3. Sample Deployment Architecture (`sample/` Folder)

- Contains a **basic Flask API prototype** for serving the trained model.
- This is **not production-ready** — it’s a **proof-of-concept for pipeline architecture**.
- Purpose:
  - Demonstrate model loading, REST API serving structure
  - Lay foundation for further development (e.g., Docker, CI/CD, monitoring)

>  **Note**: This sample code will require **refactoring and error handling** for real-world deployment.

---
##  How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd loan-data-analysis
