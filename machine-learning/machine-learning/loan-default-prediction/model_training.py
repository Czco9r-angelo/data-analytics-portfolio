"""
Model Training Pipeline for Loan Default Prediction
===================================================

This module handles:
- SMOTE (Synthetic Minority Over-sampling Technique)
- K-Nearest Neighbors (KNN) with hyperparameter tuning
- Naive Bayes with variance smoothing optimization
- Decision Tree Classifier with GridSearchCV
- Model evaluation and comparison

Author: Swithun M. Chiziko
Date: June 2022
Last Edited: December 2025
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to balance class distribution in training set.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled) - Balanced training data
    """
    print("\n" + "="*70)
    print("SMOTE - SYNTHETIC MINORITY OVER-SAMPLING")
    print("="*70)
    
    # Show original class distribution
    print("\nOriginal class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_label, count in zip(unique, counts):
        print(f"  Class {class_label}: {count:4d} samples")
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Show resampled class distribution
    print("\nResampled class distribution:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for class_label, count in zip(unique, counts):
        print(f"  Class {class_label}: {count:4d} samples")
    
    print(f"\nOriginal training samples: {len(y_train)}")
    print(f"Resampled training samples: {len(y_resampled)}")
    print(f"Synthetic samples created: {len(y_resampled) - len(y_train)}")
    
    return X_resampled, y_resampled


def visualize_class_distribution(y_train, y_test, y_val, y_resampled=None, 
                                 save_path=None):
    """
    Visualize class distribution across datasets.
    
    Parameters:
    -----------
    y_train, y_test, y_val : pd.Series
        Target variables for each split
    y_resampled : pd.Series, optional
        Resampled training target (after SMOTE)
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 4 if y_resampled is not None else 3, 
                             figsize=(16, 5))
    
    datasets = [
        ('Training Set', y_train),
        ('Validation Set', y_val),
        ('Test Set', y_test)
    ]
    
    if y_resampled is not None:
        datasets.append(('Training (SMOTE)', pd.Series(y_resampled)))
    
    for idx, (title, data) in enumerate(datasets):
        axes[idx].hist(data, bins=3, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Class', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Class Distribution Across Dataset Splits', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Class distribution plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# MODEL 1: K-NEAREST NEIGHBORS (KNN)
# ============================================================================

def train_knn_with_cv(X_train, y_train, X_test, y_test, 
                     k_values=[3, 5, 10, 20, 30, 40, 50, 60, 70, 80]):
    """
    Train KNN models with different k values and evaluate.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    k_values : list
        List of k values to test
        
    Returns:
    --------
    tuple
        (best_model, results_df, best_k)
    """
    print("\n" + "="*70)
    print("K-NEAREST NEIGHBORS (KNN) - HYPERPARAMETER TUNING")
    print("="*70)
    
    results = []
    
    for k in k_values:
        print(f"\nTesting k={k}...")
        
        # Train model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'k': k,
            'accuracy': accuracy
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Find best k
    best_idx = results_df['accuracy'].idxmax()
    best_k = results_df.loc[best_idx, 'k']
    best_accuracy = results_df.loc[best_idx, 'accuracy']
    
    print(f"\n{'='*70}")
    print(f"Best k value: {best_k}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"{'='*70}")
    
    # Train final model with best k
    best_model = KNeighborsClassifier(n_neighbors=int(best_k))
    best_model.fit(X_train, y_train)
    
    return best_model, results_df, int(best_k)


def plot_knn_performance(results_df, save_path=None):
    """
    Plot KNN accuracy vs k value.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from KNN cross-validation
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(results_df['k'], results_df['accuracy'], 
             marker='o', linewidth=2, markersize=8, color='steelblue')
    
    # Highlight best k
    best_idx = results_df['accuracy'].idxmax()
    best_k = results_df.loc[best_idx, 'k']
    best_acc = results_df.loc[best_idx, 'accuracy']
    
    plt.scatter([best_k], [best_acc], color='red', s=200, 
                zorder=5, label=f'Best k={best_k} (Acc={best_acc:.4f})')
    
    plt.xlabel('Number of Neighbors (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('KNN Performance: Accuracy vs k Value', 
              fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ KNN performance plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# MODEL 2: NAIVE BAYES
# ============================================================================

def train_naive_bayes_with_gridsearch(X_train, y_train, X_test, y_test):
    """
    Train Naive Bayes with GridSearch for var_smoothing optimization.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
        
    Returns:
    --------
    tuple
        (best_model, cv_results_df, best_params)
    """
    print("\n" + "="*70)
    print("NAIVE BAYES - VARIANCE SMOOTHING OPTIMIZATION")
    print("="*70)
    
    # Define parameter grid
    param_grid = {
        'var_smoothing': [1e+00, 1e-01, 1e-02, 1e-03, 1e-04, 
                         1e-05, 1e-06, 1e-07, 1e-08, 1e-09]
    }
    
    # Initialize model
    nb = GaussianNB()
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=nb,
        param_grid=param_grid,
        cv=10,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    print("\nFitting 10 folds for each of 10 candidates (100 fits total)...")
    grid_search.fit(X_train, y_train)
    
    # Get results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    
    # Test set performance
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Create results dataframe
    cv_results_df = pd.DataFrame({
        'var_smoothing': [p['var_smoothing'] for p in grid_search.cv_results_['params']],
        'mean_cv_score': grid_search.cv_results_['mean_test_score']
    })
    
    return grid_search.best_estimator_, cv_results_df, best_params


def plot_naive_bayes_performance(cv_results_df, save_path=None):
    """
    Plot Naive Bayes performance vs var_smoothing.
    
    Parameters:
    -----------
    cv_results_df : pd.DataFrame
        Cross-validation results
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(cv_results_df['var_smoothing'], cv_results_df['mean_cv_score'],
             marker='o', linewidth=2, markersize=8, color='coral')
    
    plt.xlabel('Variance Smoothing', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Cross-Validation Score', fontsize=12, fontweight='bold')
    plt.title('Naive Bayes Performance vs Variance Smoothing',
              fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Naive Bayes performance plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# MODEL 3: DECISION TREE
# ============================================================================

def train_decision_tree_with_gridsearch(X_train, y_train, X_test, y_test):
    """
    Train Decision Tree with GridSearch for hyperparameter optimization.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
        
    Returns:
    --------
    tuple
        (best_model, cv_results_df, best_params)
    """
    print("\n" + "="*70)
    print("DECISION TREE - HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Define parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50],
        'min_samples_split': [2, 3]
    }
    
    # Initialize model
    dt = DecisionTreeClassifier(random_state=42)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=10,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    print("\nFitting 10 folds for each of 56 candidates (560 fits total)...")
    grid_search.fit(X_train, y_train)
    
    # Get results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    
    # Test set performance
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Create results dataframe
    cv_results_df = pd.DataFrame(grid_search.cv_results_['params'])
    cv_results_df['mean_cv_score'] = grid_search.cv_results_['mean_test_score']
    
    return grid_search.best_estimator_, cv_results_df, best_params


def plot_decision_tree_performance(cv_results_df, save_path=None):
    """
    Plot Decision Tree performance: Gini vs Entropy by max_depth.
    
    Parameters:
    -----------
    cv_results_df : pd.DataFrame
        Cross-validation results
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Plot for each criterion
    for criterion in ['gini', 'entropy']:
        temp = cv_results_df[cv_results_df['criterion'] == criterion]
        # Average across min_samples_split
        temp_avg = temp.groupby('max_depth')['mean_cv_score'].mean()
        plt.plot(temp_avg.index, temp_avg.values, 
                marker='o', linewidth=2, markersize=6, label=criterion)
    
    plt.xlabel('Max Depth', fontsize=12, fontweight='bold')
    plt.ylabel('Mean CV Score', fontsize=12, fontweight='bold')
    plt.title('Decision Tree CV Performance: Gini vs Entropy',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Decision Tree performance plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation with metrics and confusion matrix.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test, y_test : Test data
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*70)
    print(f"{model_name} - EVALUATION")
    print("="*70)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='black')
    
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} Confusion Matrix\nAccuracy: {cm.trace()/cm.sum():.4f}',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models side-by-side.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model}
    X_test, y_test : Test data
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    results = []
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Confusion Matrix': cm
        })
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Sort by accuracy
    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    
    print("\n" + "="*70)
    print("RANKING (by Accuracy):")
    for idx, row in results_df.iterrows():
        print(f"  {row['Model']:25} | {row['Accuracy']:.4f}")
    
    return results_df


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    filepath : str
        Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


# ============================================================================
# COMPLETE TRAINING PIPELINE
# ============================================================================

def complete_training_pipeline(X_train, y_train, X_test, y_test, X_val, y_val):
    """
    Complete model training pipeline with all three models.
    
    Returns:
    --------
    dict
        Dictionary containing all trained models and results
    """
    # 1. Apply SMOTE
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    
    # 2. Visualize class distribution
    visualize_class_distribution(
        y_train, y_test, y_val, y_resampled,
        save_path='results/class_distribution.png'
    )
    
    # 3. Train KNN
    knn_model, knn_results, best_k = train_knn_with_cv(
        X_resampled, y_resampled, X_test, y_test
    )
    plot_knn_performance(knn_results, save_path='results/knn_performance.png')
    
    # 4. Train Naive Bayes
    nb_model, nb_results, nb_params = train_naive_bayes_with_gridsearch(
        X_resampled, y_resampled, X_test, y_test
    )
    plot_naive_bayes_performance(nb_results, save_path='results/nb_performance.png')
    
    # 5. Train Decision Tree
    dt_model, dt_results, dt_params = train_decision_tree_with_gridsearch(
        X_resampled, y_resampled, X_test, y_test
    )
    plot_decision_tree_performance(dt_results, save_path='results/dt_performance.png')
    
    # 6. Compare models
    models = {
        'KNN': knn_model,
        'Naive Bayes': nb_model,
        'Decision Tree': dt_model
    }
    comparison = compare_models(models, X_test, y_test)
    
    # 7. Save models
    save_model(knn_model, 'models/knn_model.pkl')
    save_model(nb_model, 'models/naive_bayes_model.pkl')
    save_model(dt_model, 'models/decision_tree_model.pkl')
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    
    return {
        'models': models,
        'X_resampled': X_resampled,
        'y_resampled': y_resampled,
        'comparison': comparison
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Model training module ready!")
    print("Import and use in your main script.")
