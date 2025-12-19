Loan Default Prediction - Machine Learning Project
Author: Swithun M. Chiziko
Student ID: 17487140
Institution: Curtin University
Program: Master of Predictive Analytics
Course: Data Mining
Date: June 2022
📋 Project Overview
This project implements a comprehensive machine learning pipeline for loan default prediction using an obfuscated dataset. Three classification algorithms (K-Nearest Neighbors, Naive Bayes, and Decision Tree) were developed and compared, achieving up to 84.67% accuracy with KNN and 79.35% with Decision Tree (selected for production due to superior generalization).
🎯 Key Achievements

Data Preprocessing: Handled 48% missing data using MICE framework
Feature Engineering: Reduced dimensionality from 40 to 18 features (55% reduction)
Class Imbalance: Applied SMOTE to balance training data
Model Performance:

K-Nearest Neighbors: 84.67% accuracy
Decision Tree: 79.35% accuracy (SELECTED - consistent across val/test)
Naive Bayes: 70.86% accuracy



📁 Project Structure
loan-default-prediction/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── data_preprocessing.py          # Data loading, cleaning, imputation
├── feature_engineering.py         # Feature selection and scaling
├── model_training.py              # Model training with SMOTE
├── visualizations.py              # Comprehensive plotting functions
├── main_pipeline.py               # Complete pipeline orchestration
│
├── data/
│   └── Assignment2021.sqlite      # SQLite database (not included)
│
├── models/                        # Trained models (generated)
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   └── decision_tree_model.pkl
│
├── results/                       # Generated visualizations and reports
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── class_distribution.png
│   ├── knn_performance.png
│   ├── naive_bayes_performance.png
│   ├── decision_tree_performance.png
│   ├── confusion_matrices_comparison.png
│   └── model_comparison_results.csv
│
└── docs/
    └── project_report.pdf         # Complete 20-page technical report
🚀 Quick Start
Installation
bash# Clone the repository
git clone https://github.com/Czco9r-angelo/loan-default-prediction.git
cd loan-default-prediction

# Install dependencies
pip install -r requirements.txt
Running the Complete Pipeline
bash# Run the entire pipeline (preprocessing → feature engineering → training → evaluation)
python main_pipeline.py
Using Individual Modules
python# Example: Data preprocessing only
from data_preprocessing import preprocess_pipeline

data_splits = preprocess_pipeline("Assignment2021.sqlite")
python# Example: Feature engineering only
from feature_engineering import feature_engineering_pipeline

engineered_data = feature_engineering_pipeline(
    X_train, X_test, X_val, y_train,
    n_features=18
)
python# Example: Model training only
from model_training import complete_training_pipeline

results = complete_training_pipeline(
    X_train, y_train, X_test, y_test, X_val, y_val
)
📊 Pipeline Workflow
Phase 1: Data Preprocessing
Steps:

Load data from SQLite database (1,200 records, 32 variables)
Analyze missing data patterns
Impute missing values:

Att00 (0.75%): Derived using correlation relationship
Att09 (48.42%): MICE with Decision Tree regressor


Remove highly correlated features (5 pairs identified)
One-hot encode categorical variables (Att01, Att08, Att29)
Remove rare categories (<6 observations)
Split data: 70% train, 15% validation, 15% test

Key Functions:

load_data() - Load from SQLite
impute_missing_values() - MICE framework
remove_correlated_features() - Handle multicollinearity
encode_categorical_variables() - One-hot encoding
split_data() - Stratified splitting

Phase 2: Feature Engineering
Steps:

Calculate feature importance using Decision Tree
Select top 18 features (from 40 post-encoding)
Standardize numerical features (mean=0, std=1)
Generate feature importance visualizations

Key Functions:

calculate_feature_importance() - Tree-based importance
select_top_features() - Dimensionality reduction
standardize_features() - Z-score normalization

Phase 3: Model Training
Steps:

Apply SMOTE to balance classes (201/301/498 → 348/348/348)
Train K-Nearest Neighbors:

Test k values: 3, 5, 10, 20, 30, 40, 50, 60, 70, 80
Best k=40: 84.67% accuracy


Train Naive Bayes:

GridSearch for var_smoothing optimization
10-fold cross-validation (100 fits)
Best accuracy: 70.86%


Train Decision Tree:

GridSearch: criterion, max_depth, min_samples_split
10-fold cross-validation (560 fits)
Best params: entropy, max_depth=10, min_samples_split=2
Accuracy: 79.35% (consistent on val and test)


Evaluate on validation set for generalization check

Key Functions:

apply_smote() - Synthetic oversampling
train_knn_with_cv() - KNN hyperparameter tuning
train_naive_bayes_with_gridsearch() - NB optimization
train_decision_tree_with_gridsearch() - DT optimization
evaluate_model() - Comprehensive evaluation

Phase 4: Model Comparison & Export
Steps:

Generate confusion matrices for all models
Create performance comparison visualizations
Export model comparison results to CSV
Save all trained models (.pkl files)

Key Functions:

compare_models() - Side-by-side comparison
plot_confusion_matrices_comparison() - Visual comparison
save_model() - Model persistence

🔬 Methodology Highlights
Missing Value Imputation
Strategy 1: Derived Relationships

Att00 imputed using: Att00 = Att10 / 38.5662
Justified by perfect correlation (r = 1.0)

Strategy 2: MICE (Multivariate Imputation by Chained Equations)

Used for Att09 (48% missing)
Decision Tree regressor as estimator
10 iterations until convergence
Separate imputation per dataset split (prevent leakage)

Feature Selection Rationale
Correlation Analysis:

Identified 5 perfectly correlated pairs (r = 1.0):

Att10 = 38.5662 × Att00
Att18 = 4.8939 × Att03
Att11 = 3.56225 × Att04
Att07 = 0.52983 × Att05
Att28 = 0.00149 × Att22


Removed one variable from each pair

Importance-Based Selection:

Used Decision Tree Classifier (entropy, max_depth=10)
Removed 22 features with zero importance
Retained top 18 features capturing 80%+ predictive power

Class Imbalance Handling
SMOTE (Synthetic Minority Over-sampling Technique):

Original distribution: Class 0 (201), Class 1 (301), Class 2 (498)
Applied only to training set (prevent data leakage)
Post-SMOTE: Equal representation (348 each)
Validation/test sets kept original distribution

Model Selection Decision
Why Decision Tree was selected over KNN despite lower accuracy:

Consistency: 79.35% on both validation AND test sets

No signs of overfitting
Reliable generalization expected


KNN Concern: 84.67% accuracy but potential overfitting

Higher variance in cross-validation
May not generalize as well to production data


Production Benefits:

Interpretable (feature importance, tree structure)
Fast inference (no distance calculations)
Stable performance across datasets



📈 Results Summary
Model Performance
ModelTest AccuracyValidation AccuracySelectedDecision Tree79.35%79.35%✓K-Nearest Neighbors84.67%-✗Naive Bayes70.86%-✗
Decision Tree Confusion Matrix
                Predicted
                0    1    2
Actual    0    16    5    4    (64% recall)
          1     7   27   11    (60% recall)
          2     8    5   67    (84% recall)

Overall Accuracy: 79.35%
Top 5 Most Important Features

Att24: 12.84% importance
Att15: 10.89% importance
Att26: 9.90% importance
Att28: 9.32% importance
Att14: 7.37% importance

📚 Technical Stack
Core Libraries:
pandas       - Data manipulation
numpy        - Numerical computing
scikit-learn - Machine learning algorithms
imbalanced-learn - SMOTE implementation
Visualization:
matplotlib   - Plotting
seaborn      - Statistical visualizations
scipy        - Statistical tests
Database:
sqlite3      - Database connectivity (built-in)
🎓 Academic Context
This project was completed as part of a Master's level Data Mining course at Curtin University. The dataset was intentionally obfuscated to simulate real-world scenarios where domain knowledge is limited, requiring purely data-driven techniques for feature engineering and model selection.
Learning Objectives Demonstrated:

Advanced data preprocessing and imputation
Handling multicollinearity and feature selection
Addressing class imbalance
Hyperparameter optimization via GridSearchCV
Model evaluation and selection criteria
Production-ready code organization

📝 Code Quality Features
✅ Comprehensive Documentation

Detailed docstrings for all functions
Inline comments explaining complex logic
README with usage examples

✅ Modular Design

Separate modules for each pipeline stage
Reusable functions
Easy to test and maintain

✅ Best Practices

Type hints where appropriate
Consistent naming conventions
Error handling
Configuration management

✅ Reproducibility

Random seeds for all stochastic operations
Complete requirements.txt
Clear pipeline execution order

🔧 Customization
Changing Hyperparameters
Edit main_pipeline.py Config class:
pythonclass Config:
    N_FEATURES = 18  # Change number of features
    KNN_K_VALUES = [3, 5, 10, 20, 30, 40]  # Modify k values
    TEST_SIZE = 0.3  # Adjust test split
    # ... other parameters
Using Different Models
Add to model_training.py:
pythonfrom sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf
📧 Contact
Swithun M. Chiziko

Email: chizikoswith@gmail.com
LinkedIn: linkedin.com/in/swithun-chiziko-94a21869
GitHub: github.com/swithun-chiziko

📄 License
This project is for academic and portfolio purposes. The dataset is confidential and not included in this repository. Code is available under MIT License.
🙏 Acknowledgments

Curtin University - Data Mining course and dataset
scikit-learn - Machine learning library
imbalanced-learn - SMOTE implementation
Academic references - See project report for full citation list


Last Updated: December 2025
Project Status: ✅ Completed | 📊 Academic Portfolio | 🎓 Master's Level Work
