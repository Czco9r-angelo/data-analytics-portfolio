# Machine Learning Projects

This folder contains machine learning and predictive analytics projects demonstrating classification, regression, clustering, and forecasting techniques.

## Projects Overview

### 1. Loan Default Prediction
**Objective:** Predict probability of loan default using obfuscated historical borrower data without the context of understandable variable names

**Approach:**
- Data preprocessing and feature engineering
- Multiple algorithms tested (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting)
- Model evaluation using accuracy, precision, recall, and ROC-AUC
- Final model: Random Forest with ~80% accuracy

**Technologies:** Python, pandas, scikit-learn, matplotlib

📁 [View Project Details](./loan-default-prediction/)

---

### 2. Stock Momentum Analysis
**Objective:** Classify stocks based on earnings momentum and price patterns

**Approach:**
- Feature engineering from financial statements and price data
- Classification models for momentum scoring
- Backtesting and performance validation
- Integration with investment strategy

**Technologies:** Python, pandas, NumPy, scikit-learn

📁 [View Project Details](./stock-momentum-analysis/)

---

### 3. Risk Modeling Portfolio
**Objective:** Quantitative risk assessment using Monte Carlo simulation

**Approach:**
- Portfolio value-at-risk (VaR) calculation
- Scenario analysis and sensitivity testing
- Probability distributions for return forecasting
- Downside risk quantification

**Technologies:** Python, NumPy, scipy, matplotlib

📁 [View Project Details](./risk-modeling/)

---

## Technical Skills Demonstrated

- **Data Preprocessing:** Handling missing values, outlier detection, feature scaling
- **Feature Engineering:** Creating derived features, dimensionality reduction
- **Model Selection:** Comparing multiple algorithms, hyperparameter tuning
- **Model Evaluation:** Cross-validation, confusion matrices, ROC curves
- **Interpretation:** Feature importance, SHAP values, business insights

## Tools & Libraries
```python
# Core libraries used across projects
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

## Contact

Questions about these projects? Reach out:
- Email: chizikoswith@gmail.com
- LinkedIn: [Swithun Chiziko](https://linkedin.com/in/swithun-chiziko-94a21869)
