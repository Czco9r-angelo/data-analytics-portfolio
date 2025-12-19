"""
Data Preprocessing Pipeline for Loan Default Prediction
========================================================

This module handles:
- Data loading from SQLite database
- Exploratory data analysis
- Missing value imputation (MICE framework)
- Correlation analysis
- Data splitting
- Categorical encoding
- Rare category removal

Author: Swithun M. Chiziko
Date: June 2022
Last Edited: December 2025
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(database_path="Assignment2021.sqlite"):
    """
    Load data from SQLite database.
    
    Parameters:
    -----------
    database_path : str
        Path to the SQLite database file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    print(f"Loading data from {database_path}...")
    
    # Connect to database
    conn = sqlite3.connect(database_path)
    
    # Query all data
    query = "SELECT * FROM DATA;"
    df = pd.read_sql_query(query, conn)
    
    # Close connection
    conn.close()
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

def analyze_missing_data(df):
    """
    Analyze and report missing data in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing data by column
    """
    print("\n" + "="*70)
    print("MISSING DATA ANALYSIS")
    print("="*70)
    
    missing_summary = []
    
    for column in df.columns:
        missing_count = df[column].isna().sum()
        total_count = len(df)
        missing_pct = (missing_count / total_count) * 100
        
        if missing_count > 0:
            print(f"{column:15} | {missing_count:4d}/{total_count:4d} missing | {missing_pct:6.2f}%")
            missing_summary.append({
                'column': column,
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            })
    
    if not missing_summary:
        print("No missing data found!")
    
    return pd.DataFrame(missing_summary)


def analyze_categorical_distribution(df, categorical_columns):
    """
    Analyze distribution of categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_columns : list
        List of categorical column names
        
    Returns:
    --------
    dict
        Distribution summary for each categorical column
    """
    print("\n" + "="*70)
    print("CATEGORICAL VARIABLE DISTRIBUTION")
    print("="*70)
    
    distributions = {}
    
    for column in categorical_columns:
        print(f"\n{column}:")
        value_counts = df[column].value_counts()
        print(value_counts)
        distributions[column] = value_counts
        
        # Identify rare categories (< 6 observations)
        rare_categories = value_counts[value_counts < 6]
        if len(rare_categories) > 0:
            print(f"  ⚠️  Rare categories (< 6 obs): {list(rare_categories.index)}")
    
    return distributions


def analyze_class_distribution(df, target_column='class'):
    """
    Analyze distribution of target variable (class imbalance).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of target column
        
    Returns:
    --------
    pd.Series
        Class distribution
    """
    print("\n" + "="*70)
    print("TARGET VARIABLE (CLASS) DISTRIBUTION")
    print("="*70)
    
    class_dist = df[target_column].value_counts().sort_index()
    
    for class_label, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"Class {class_label}: {count:4d} observations ({percentage:5.2f}%)")
    
    # Calculate imbalance ratio
    imbalance_ratio = class_dist.max() / class_dist.min()
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
    
    return class_dist


def analyze_correlations(df, threshold=0.95):
    """
    Identify highly correlated variable pairs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    threshold : float
        Correlation threshold for identifying high correlation
        
    Returns:
    --------
    list
        List of highly correlated variable pairs
    """
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                # Check if relationship is derivable
                if not df[col1].isna().any() and not df[col2].isna().any():
                    quotient = (df[col2] / df[col1]).mean()
                    
                    print(f"\n{col1} ↔ {col2}")
                    print(f"  Correlation: {corr_value:.4f}")
                    print(f"  Quotient ({col2}/{col1}): {quotient:.6f}")
                    
                    high_corr_pairs.append({
                        'var1': col1,
                        'var2': col2,
                        'correlation': corr_value,
                        'quotient': quotient
                    })
    
    return high_corr_pairs


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def impute_missing_values(df, correlated_pairs):
    """
    Impute missing values using:
    1. Derived relationships for highly correlated variables
    2. MICE framework for other missing data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with missing values
    correlated_pairs : list
        List of correlated variable pairs from correlation analysis
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with imputed values
    """
    print("\n" + "="*70)
    print("MISSING VALUE IMPUTATION")
    print("="*70)
    
    df_imputed = df.copy()
    
    # 1. Impute Att00 using relationship with Att10
    # Att00 = Att10 / 38.5662
    if 'Att00' in df.columns and df['Att00'].isna().any():
        missing_count = df['Att00'].isna().sum()
        print(f"\nImputing Att00 using relationship: Att00 = Att10 / 38.5662")
        print(f"  Missing values: {missing_count}")
        
        df_imputed.loc[df_imputed['Att00'].isna(), 'Att00'] = \
            df_imputed.loc[df_imputed['Att00'].isna(), 'Att10'] / 38.5662
        
        print(f"  ✓ Imputation complete")
    
    # 2. MICE imputation for Att09 (48% missing)
    if 'Att09' in df.columns and df['Att09'].isna().any():
        missing_count = df['Att09'].isna().sum()
        print(f"\nImputing Att09 using MICE (Decision Tree Regressor)")
        print(f"  Missing values: {missing_count} ({missing_count/len(df)*100:.1f}%)")
        
        # Select numerical columns for imputation
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
        
        # Initialize MICE imputer with Decision Tree
        imputer = IterativeImputer(
            estimator=DecisionTreeRegressor(max_depth=10, random_state=42),
            max_iter=10,
            random_state=42
        )
        
        # Fit and transform
        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
        
        print(f"  ✓ MICE imputation complete (10 iterations)")
    
    # Verify no missing values remain
    remaining_missing = df_imputed.isna().sum().sum()
    print(f"\nRemaining missing values: {remaining_missing}")
    
    return df_imputed


def split_data(df, test_size=0.3, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Strategy:
    - 70% training
    - 15% validation (from training split)
    - 30% test
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set (from remaining after test split)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing train, validation, and test dataframes
    """
    print("\n" + "="*70)
    print("DATA SPLITTING")
    print("="*70)
    
    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"Training set:   {len(X_train):4d} samples ({len(X_train)/len(df)*100:5.1f}%)")
    print(f"Validation set: {len(X_val):4d} samples ({len(X_val)/len(df)*100:5.1f}%)")
    print(f"Test set:       {len(X_test):4d} samples ({len(X_test)/len(df)*100:5.1f}%)")
    
    # Check class distribution in each split
    print("\nClass distribution:")
    print(f"  Train: {dict(y_train.value_counts().sort_index())}")
    print(f"  Val:   {dict(y_val.value_counts().sort_index())}")
    print(f"  Test:  {dict(y_test.value_counts().sort_index())}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def remove_correlated_features(df, features_to_remove):
    """
    Remove one variable from each highly correlated pair.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features_to_remove : list
        List of feature names to remove
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with correlated features removed
    """
    print("\n" + "="*70)
    print("REMOVING HIGHLY CORRELATED FEATURES")
    print("="*70)
    
    print(f"Original dimensions: {df.shape}")
    print(f"Features to remove: {features_to_remove}")
    
    df_reduced = df.drop(columns=features_to_remove, errors='ignore')
    
    print(f"New dimensions: {df_reduced.shape}")
    print(f"Features removed: {df.shape[1] - df_reduced.shape[1]}")
    
    return df_reduced


def encode_categorical_variables(df, categorical_columns):
    """
    One-hot encode categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_columns : list
        List of categorical column names to encode
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with one-hot encoded categorical variables
    """
    print("\n" + "="*70)
    print("ONE-HOT ENCODING CATEGORICAL VARIABLES")
    print("="*70)
    
    print(f"Original dimensions: {df.shape}")
    print(f"Columns to encode: {categorical_columns}")
    
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    
    print(f"New dimensions: {df_encoded.shape}")
    print(f"New columns created: {df_encoded.shape[1] - df.shape[1]}")
    
    return df_encoded


def remove_rare_categories(df, rare_category_columns):
    """
    Remove rare category columns (< 6 observations).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    rare_category_columns : list
        List of rare category column names to remove
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with rare categories removed
    """
    print("\n" + "="*70)
    print("REMOVING RARE CATEGORY COLUMNS")
    print("="*70)
    
    print(f"Rare categories to remove: {rare_category_columns}")
    
    df_cleaned = df.drop(columns=rare_category_columns, errors='ignore')
    
    print(f"Columns removed: {len(rare_category_columns)}")
    print(f"Final dimensions: {df_cleaned.shape}")
    
    return df_cleaned


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_pipeline(database_path="Assignment2021.sqlite"):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    database_path : str
        Path to SQLite database
        
    Returns:
    --------
    dict
        Dictionary containing processed train/val/test sets
    """
    # 1. Load data
    df = load_data(database_path)
    
    # 2. Exploratory analysis
    analyze_missing_data(df)
    analyze_categorical_distribution(df, ['Att01', 'Att08', 'Att29'])
    analyze_class_distribution(df)
    corr_pairs = analyze_correlations(df, threshold=0.95)
    
    # 3. Impute missing values
    df_imputed = impute_missing_values(df, corr_pairs)
    
    # 4. Remove highly correlated features
    features_to_remove = ['Att05', 'Att00', 'Att11', 'Att18', 'Att22']
    df_reduced = remove_correlated_features(df_imputed, features_to_remove)
    
    # 5. Encode categorical variables
    df_encoded = encode_categorical_variables(
        df_reduced, 
        ['Att01', 'Att08', 'Att29']
    )
    
    # 6. Remove rare categories
    rare_categories = [
        'Att01_ACKH', 'Att01_TRRP', 'Att01_UJJW', 
        'Att08_VEVT', 'Att29_PJIY'
    ]
    df_final = remove_rare_categories(df_encoded, rare_categories)
    
    # 7. Split data
    splits = split_data(df_final)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    
    return splits


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run complete preprocessing pipeline
    data_splits = preprocess_pipeline()
    
    print("\nPreprocessed data ready for modeling!")
    print(f"Training set shape: {data_splits['X_train'].shape}")

