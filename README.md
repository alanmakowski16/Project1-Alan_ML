# 🚀 Adult Census Income Prediction

This project is a complete data science workflow to build a machine learning model that predicts whether an individual's income exceeds $50,000 per year, based on 1994 US Census data.

The primary goal is to optimize for the **F1-Macro score** due to the significant class imbalance in the dataset.

## 🎯 Project Workflow

This repository contains a single Jupyter Notebook that follows these steps:

### 1. Data Cleaning
* Loaded the raw data and immediately cleaned the target variable (`income`) by removing erroneous trailing dots (`.`).
* Built a robust `sklearn.pipeline.Pipeline` with custom transformers to:
    * Standardize all missing values (e.g., `'?'`) to a consistent `'missing'` category.
    * Strip whitespace and lowercase all string features for consistency.

### 2. Exploratory Data Analysis (EDA)
* Analyzed the target variable, confirming a **~76% / 24% class imbalance**.
* Plotted numerical features (like `age`, `hours-per-week`) against income using KDE plots to visualize distributions.
* Plotted categorical features (like `occupation`, `marital-status`) using normalized bar charts to identify the strongest predictors of high income.

### 3. Advanced Preprocessing
* **Split Data:** The data was split into training and testing sets using `train_test_split` with `stratify=y` to ensure the class imbalance was preserved in both sets.
* **Target Encoding `y`:** The target labels (`<=50k`, `>50k`) were `LabelEncoder` encoded to `0` and `1` for model compatibility.
* **Feature Engineering Pipeline:** A `ColumnTransformer` was built to apply different, optimized strategies to each column type:
    * **High-Cardinality Features** (`occupation`, `native-country`): Used `TargetEncoder` to encode these features based on their correlation with the target, avoiding the "curse of dimensionality."
    * **Low-Cardinality Features** (`workclass`, `sex`, `race`, etc.): Used `OneHotEncoder` to create binary columns.
    * **Numerical Features:** Scaled using `StandardScaler`.
    * **Dropped Features:** `fnlwgt` (a statistical weight) and `education` (redundant) were dropped.

### 4. Model Training & Hyperparameter Tuning
* **Imbalance Handling:** We used algorithmic-level approaches:
    * `RandomForestClassifier`: `class_weight='balanced'`
    * `XGBClassifier`: `scale_pos_weight` (calculated as `count(neg) / count(pos)`)
* **Tuning with Optuna:**
    * We used **Optuna** instead of a simple GridSearch for more efficient and intelligent hyperparameter tuning.
    * The Optuna `objective` function was built to use 5-fold `StratifiedKFold` cross-validation, optimizing for the `f1_macro` score to get a stable performance metric.

### 5. Model Evaluation & Refinement
* **Model Comparison:** The tuned `RandomForestClassifier` (baseline) and `XGBClassifier` (advanced) were compared.
* **Decision Threshold Tuning:** After training the final XGBoost model, we analyzed its probability outputs (`.predict_proba()`) and used the `precision_recall_curve` to find a new decision threshold (e.g., ~0.4) that maximized the F1-score for the minority class (`>50k`), further improving performance.
* **Feature Importance:** Plotted the final feature importances from the XGBoost model to understand what features drove its predictions.

## 🛠️ Key Libraries & Techniques
* **Data Processing:** Pandas, Scikit-learn (`Pipeline`, `ColumnTransformer`, `TargetEncoder`, `OneHotEncoder`, `StandardScaler`)
* **Modeling:** `RandomForestClassifier`, `XGBClassifier`
* **Tuning:** Optuna
* **Imbalance Handling:** `stratify`, `scale_pos_weight`, `class_weight`
* **Evaluation:** `classification_report`, `f1_score`, `precision_recall_curve`, `roc_auc_score`
