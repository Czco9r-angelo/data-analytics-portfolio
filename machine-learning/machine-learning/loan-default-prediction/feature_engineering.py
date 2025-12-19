"""
Feature Engineering and Selection for Loan Default Prediction
==============================================================

This module handles:
- Feature scaling (standardization)
- Feature importance analysis
- Feature selection based on importance scores
- Dimensionality reduction

Author: Swithun M. Chiziko
Date: June 2022
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# FEATURE SCALING
# ============================================================================

def standardize_features(X_train, X_test, X_val=None):
    """
    Standardize numerical features using training set statistics.
    
    Important: Use training set mean/std to scale test and validation sets
    to prevent data leakage.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    X_val : pd.DataFrame, optional
        Validation features
        
    Returns:
    --------
    dict
        Dictionary containing standardized dataframes
    """
    print("\n" + "="*70)
    print("FEATURE STANDARDIZATION")
    print("="*70)
    
    # Identify numerical columns (float64)
    numerical_cols = X_train.select_dtypes(include='float64').columns
    print(f"Numerical columns to standardize: {len(numerical_cols)}")
    
    # Create copies to avoid modifying originals
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()
    X_val_std = X_val.copy() if X_val is not None else None
    
    # Standardize using training set statistics
    for col in numerical_cols:
        train_mean = X_train[col].mean()
        train_std = X_train[col].std()
        
        # Standardize training set
        X_train_std[col] = (X_train[col] - train_mean) / train_std
        
        # Standardize test set using TRAINING statistics
        X_test_std[col] = (X_test[col] - train_mean) / train_std
        
        # Standardize validation set using TRAINING statistics
        if X_val is not None:
            X_val_std[col] = (X_val[col] - train_mean) / train_std
    
    print("✓ Standardization complete")
    print(f"  Training set: mean ≈ 0, std ≈ 1")
    print(f"  Test/Val sets: scaled using training statistics")
    
    result = {
        'X_train': X_train_std,
        'X_test': X_test_std
    }
    
    if X_val is not None:
        result['X_val'] = X_val_std
    
    return result


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def calculate_feature_importance(X_train, y_train, random_state=42):
    """
    Calculate feature importance using Decision Tree Classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Feature importance scores sorted by importance
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Train Decision Tree for feature importance
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=10,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': clf.feature_importances_
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(
        'importance', 
        ascending=False
    ).reset_index(drop=True)
    
    # Display results
    print(f"\nFeature importance scores:")
    print("-" * 70)
    for idx, row in feature_importance_df.iterrows():
        print(f"{row['feature']:30} | {row['importance']:.6f}")
    
    # Count zero-importance features
    zero_importance = (feature_importance_df['importance'] == 0).sum()
    print(f"\nFeatures with zero importance: {zero_importance}")
    
    return feature_importance_df


def plot_feature_importance(feature_importance_df, top_n=20, save_path=None):
    """
    Visualize feature importance scores.
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        Feature importance dataframe
    top_n : int
        Number of top features to display
    save_path : str, optional
        Path to save the figure
    """
    # Select top N features
    top_features = feature_importance_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    plt.barh(
        range(len(top_features)), 
        top_features['importance'],
        color=colors
    )
    
    # Customize plot
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(
        f'Top {top_n} Feature Importance Scores\n(Decision Tree Classifier)',
        fontsize=14,
        fontweight='bold'
    )
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_top_features(X, feature_importance_df, n_features=18):
    """
    Select top N features based on importance scores.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    feature_importance_df : pd.DataFrame
        Feature importance scores
    n_features : int
        Number of top features to select
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with selected features only
    """
    print("\n" + "="*70)
    print("FEATURE SELECTION")
    print("="*70)
    
    print(f"Original dimensions: {X.shape}")
    
    # Get top N features
    top_features = feature_importance_df.head(n_features)['feature'].tolist()
    
    # Select only these features
    X_selected = X[top_features]
    
    print(f"Selected features: {n_features}")
    print(f"New dimensions: {X_selected.shape}")
    print(f"Dimensionality reduction: {(1 - n_features/X.shape[1])*100:.1f}%")
    
    print("\nSelected features:")
    for i, feature in enumerate(top_features, 1):
        importance = feature_importance_df[
            feature_importance_df['feature'] == feature
        ]['importance'].values[0]
        print(f"  {i:2d}. {feature:30} | {importance:.6f}")
    
    return X_selected


def remove_zero_importance_features(X, feature_importance_df, threshold=0.0):
    """
    Remove features with zero or near-zero importance.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    feature_importance_df : pd.DataFrame
        Feature importance scores
    threshold : float
        Importance threshold (features below this are removed)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with zero-importance features removed
    """
    print("\n" + "="*70)
    print("REMOVING ZERO-IMPORTANCE FEATURES")
    print("="*70)
    
    print(f"Original dimensions: {X.shape}")
    
    # Identify features to keep
    features_to_keep = feature_importance_df[
        feature_importance_df['importance'] > threshold
    ]['feature'].tolist()
    
    # Select only important features
    X_reduced = X[features_to_keep]
    
    features_removed = X.shape[1] - X_reduced.shape[1]
    
    print(f"Features removed: {features_removed}")
    print(f"Features retained: {len(features_to_keep)}")
    print(f"New dimensions: {X_reduced.shape}")
    
    return X_reduced


# ============================================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================================

def feature_engineering_pipeline(X_train, X_test, X_val, y_train, 
                                 n_features=18, plot_importance=True):
    """
    Complete feature engineering pipeline.
    
    Steps:
    1. Calculate feature importance
    2. Select top N features
    3. Standardize features
    4. Generate visualizations
    
    Parameters:
    -----------
    X_train, X_test, X_val : pd.DataFrame
        Feature dataframes
    y_train : pd.Series
        Training target
    n_features : int
        Number of features to select
    plot_importance : bool
        Whether to generate importance plot
        
    Returns:
    --------
    dict
        Dictionary containing processed features and importance scores
    """
    # 1. Calculate feature importance
    feature_importance_df = calculate_feature_importance(X_train, y_train)
    
    # 2. Plot feature importance
    if plot_importance:
        plot_feature_importance(
            feature_importance_df, 
            top_n=min(20, len(feature_importance_df)),
            save_path='results/feature_importance.png'
        )
    
    # 3. Select top features
    X_train_selected = select_top_features(X_train, feature_importance_df, n_features)
    X_test_selected = select_top_features(X_test, feature_importance_df, n_features)
    X_val_selected = select_top_features(X_val, feature_importance_df, n_features)
    
    # 4. Standardize features
    standardized = standardize_features(
        X_train_selected, 
        X_test_selected, 
        X_val_selected
    )
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)
    
    return {
        'X_train': standardized['X_train'],
        'X_test': standardized['X_test'],
        'X_val': standardized['X_val'],
        'feature_importance': feature_importance_df,
        'selected_features': X_train_selected.columns.tolist()
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Load preprocessed data and apply feature engineering
    from data_preprocessing import preprocess_pipeline
    
    # Load and preprocess data
    data = preprocess_pipeline()
    
    # Apply feature engineering
    engineered_data = feature_engineering_pipeline(
        X_train=data['X_train'],
        X_test=data['X_test'],
        X_val=data['X_val'],
        y_train=data['y_train'],
        n_features=18,
        plot_importance=True
    )
    
    print("\nFeature engineering complete!")
    print(f"Final training set shape: {engineered_data['X_train'].shape}")
