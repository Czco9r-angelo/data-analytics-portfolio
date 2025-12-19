"""
Main Pipeline for Loan Default Prediction
==========================================

This script runs the complete machine learning pipeline:
1. Data loading and preprocessing
2. Exploratory data analysis with visualizations
3. Feature engineering and selection
4. Model training with SMOTE
5. Model evaluation and comparison
6. Results export

Author: Swithun M. Chiziko
Student ID: 17487140
Institution: Curtin University
Date: June 2022
Last Edited: December 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
from data_preprocessing import (
    load_data,
    analyze_missing_data,
    analyze_categorical_distribution,
    analyze_class_distribution,
    analyze_correlations,
    impute_missing_values,
    split_data,
    remove_correlated_features,
    encode_categorical_variables,
    remove_rare_categories
)

from feature_engineering import (
    calculate_feature_importance,
    plot_feature_importance,
    select_top_features,
    standardize_features
)

from model_training import (
    apply_smote,
    visualize_class_distribution,
    train_knn_with_cv,
    plot_knn_performance,
    train_naive_bayes_with_gridsearch,
    plot_naive_bayes_performance,
    train_decision_tree_with_gridsearch,
    plot_decision_tree_performance,
    evaluate_model,
    plot_confusion_matrix,
    compare_models,
    save_model
)

from visualizations import (
    plot_diagnostic_normality,
    plot_correlation_heatmap,
    plot_confusion_matrices_comparison,
    plot_model_comparison_bar
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the pipeline."""
    
    # Paths
    DATABASE_PATH = "Assignment2021.sqlite"
    RESULTS_DIR = "results"
    MODELS_DIR = "models"
    
    # Data splitting
    TEST_SIZE = 0.3
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Feature selection
    N_FEATURES = 18
    CORRELATION_THRESHOLD = 0.95
    
    # Model parameters
    KNN_K_VALUES = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    
    # Features to remove (highly correlated)
    CORRELATED_FEATURES_TO_REMOVE = ['Att05', 'Att00', 'Att11', 'Att18', 'Att22']
    
    # Categorical features
    CATEGORICAL_FEATURES = ['Att01', 'Att08', 'Att29']
    
    # Rare categories to remove
    RARE_CATEGORIES = [
        'Att01_ACKH', 'Att01_TRRP', 'Att01_UJJW', 
        'Att08_VEVT', 'Att29_PJIY'
    ]


def setup_directories():
    """Create necessary directories for outputs."""
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(f"{Config.RESULTS_DIR}/diagnostics", exist_ok=True)
    print("✓ Output directories created")


# ============================================================================
# PHASE 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def phase1_data_preprocessing():
    """
    Phase 1: Load and preprocess data.
    
    Returns:
    --------
    dict
        Dictionary containing preprocessed data splits
    """
    print("\n" + "="*80)
    print("PHASE 1: DATA LOADING AND PREPROCESSING")
    print("="*80)
    
    # Load data
    df = load_data(Config.DATABASE_PATH)
    
    # Exploratory analysis
    analyze_missing_data(df)
    analyze_categorical_distribution(df, Config.CATEGORICAL_FEATURES)
    analyze_class_distribution(df)
    corr_pairs = analyze_correlations(df, threshold=Config.CORRELATION_THRESHOLD)
    
    # Generate correlation heatmap
    plot_correlation_heatmap(
        df.drop('class', axis=1, errors='ignore'),
        figsize=(20, 16),
        save_path=f"{Config.RESULTS_DIR}/correlation_heatmap.png"
    )
    
    # Impute missing values
    df_imputed = impute_missing_values(df, corr_pairs)
    
    # Remove highly correlated features
    df_reduced = remove_correlated_features(
        df_imputed, 
        Config.CORRELATED_FEATURES_TO_REMOVE
    )
    
    # Encode categorical variables
    df_encoded = encode_categorical_variables(
        df_reduced,
        Config.CATEGORICAL_FEATURES
    )
    
    # Remove rare categories
    df_final = remove_rare_categories(df_encoded, Config.RARE_CATEGORIES)
    
    # Split data
    data_splits = split_data(
        df_final,
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE
    )
    
    print("\n✓ Phase 1 Complete")
    return data_splits


# ============================================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================================

def phase2_feature_engineering(data_splits):
    """
    Phase 2: Feature engineering and selection.
    
    Parameters:
    -----------
    data_splits : dict
        Data splits from Phase 1
        
    Returns:
    --------
    dict
        Dictionary containing engineered features
    """
    print("\n" + "="*80)
    print("PHASE 2: FEATURE ENGINEERING AND SELECTION")
    print("="*80)
    
    # Calculate feature importance
    feature_importance_df = calculate_feature_importance(
        data_splits['X_train'],
        data_splits['y_train']
    )
    
    # Plot feature importance
    plot_feature_importance(
        feature_importance_df,
        top_n=20,
        save_path=f"{Config.RESULTS_DIR}/feature_importance.png"
    )
    
    # Select top features
    X_train_selected = select_top_features(
        data_splits['X_train'],
        feature_importance_df,
        n_features=Config.N_FEATURES
    )
    X_test_selected = select_top_features(
        data_splits['X_test'],
        feature_importance_df,
        n_features=Config.N_FEATURES
    )
    X_val_selected = select_top_features(
        data_splits['X_val'],
        feature_importance_df,
        n_features=Config.N_FEATURES
    )
    
    # Standardize features
    standardized = standardize_features(
        X_train_selected,
        X_test_selected,
        X_val_selected
    )
    
    print("\n✓ Phase 2 Complete")
    
    return {
        'X_train': standardized['X_train'],
        'X_test': standardized['X_test'],
        'X_val': standardized['X_val'],
        'y_train': data_splits['y_train'],
        'y_test': data_splits['y_test'],
        'y_val': data_splits['y_val'],
        'feature_importance': feature_importance_df
    }


# ============================================================================
# PHASE 3: MODEL TRAINING
# ============================================================================

def phase3_model_training(engineered_data):
    """
    Phase 3: Train and evaluate all models.
    
    Parameters:
    -----------
    engineered_data : dict
        Engineered features from Phase 2
        
    Returns:
    --------
    dict
        Dictionary containing trained models and results
    """
    print("\n" + "="*80)
    print("PHASE 3: MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    # Apply SMOTE
    X_resampled, y_resampled = apply_smote(
        engineered_data['X_train'],
        engineered_data['y_train']
    )
    
    # Visualize class distribution
    visualize_class_distribution(
        engineered_data['y_train'],
        engineered_data['y_test'],
        engineered_data['y_val'],
        y_resampled,
        save_path=f"{Config.RESULTS_DIR}/class_distribution.png"
    )
    
    # Train KNN
    print("\n" + "-"*80)
    knn_model, knn_results, best_k = train_knn_with_cv(
        X_resampled,
        y_resampled,
        engineered_data['X_test'],
        engineered_data['y_test'],
        k_values=Config.KNN_K_VALUES
    )
    plot_knn_performance(
        knn_results,
        save_path=f"{Config.RESULTS_DIR}/knn_performance.png"
    )
    knn_eval = evaluate_model(
        knn_model,
        engineered_data['X_test'],
        engineered_data['y_test'],
        model_name="K-Nearest Neighbors"
    )
    
    # Train Naive Bayes
    print("\n" + "-"*80)
    nb_model, nb_results, nb_params = train_naive_bayes_with_gridsearch(
        X_resampled,
        y_resampled,
        engineered_data['X_test'],
        engineered_data['y_test']
    )
    plot_naive_bayes_performance(
        nb_results,
        save_path=f"{Config.RESULTS_DIR}/naive_bayes_performance.png"
    )
    nb_eval = evaluate_model(
        nb_model,
        engineered_data['X_test'],
        engineered_data['y_test'],
        model_name="Naive Bayes"
    )
    
    # Train Decision Tree
    print("\n" + "-"*80)
    dt_model, dt_results, dt_params = train_decision_tree_with_gridsearch(
        X_resampled,
        y_resampled,
        engineered_data['X_test'],
        engineered_data['y_test']
    )
    plot_decision_tree_performance(
        dt_results,
        save_path=f"{Config.RESULTS_DIR}/decision_tree_performance.png"
    )
    dt_eval = evaluate_model(
        dt_model,
        engineered_data['X_test'],
        engineered_data['y_test'],
        model_name="Decision Tree"
    )
    
    # Validation set performance
    print("\n" + "-"*80)
    print("VALIDATION SET PERFORMANCE")
    print("-"*80)
    
    dt_val_eval = evaluate_model(
        dt_model,
        engineered_data['X_val'],
        engineered_data['y_val'],
        model_name="Decision Tree (Validation)"
    )
    
    print("\n✓ Phase 3 Complete")
    
    return {
        'models': {
            'KNN': knn_model,
            'Naive Bayes': nb_model,
            'Decision Tree': dt_model
        },
        'evaluations': {
            'KNN': knn_eval,
            'Naive Bayes': nb_eval,
            'Decision Tree': dt_eval,
            'Decision Tree (Val)': dt_val_eval
        },
        'X_resampled': X_resampled,
        'y_resampled': y_resampled
    }


# ============================================================================
# PHASE 4: MODEL COMPARISON AND EXPORT
# ============================================================================

def phase4_comparison_and_export(training_results, engineered_data):
    """
    Phase 4: Compare models and export results.
    
    Parameters:
    -----------
    training_results : dict
        Results from Phase 3
    engineered_data : dict
        Engineered features from Phase 2
    """
    print("\n" + "="*80)
    print("PHASE 4: MODEL COMPARISON AND RESULTS EXPORT")
    print("="*80)
    
    # Compare models
    comparison_df = compare_models(
        training_results['models'],
        engineered_data['X_test'],
        engineered_data['y_test']
    )
    
    # Plot confusion matrices comparison
    cms = [
        training_results['evaluations']['KNN']['confusion_matrix'],
        training_results['evaluations']['Naive Bayes']['confusion_matrix'],
        training_results['evaluations']['Decision Tree']['confusion_matrix']
    ]
    accuracies = [
        training_results['evaluations']['KNN']['accuracy'],
        training_results['evaluations']['Naive Bayes']['accuracy'],
        training_results['evaluations']['Decision Tree']['accuracy']
    ]
    
    plot_confusion_matrices_comparison(
        cms,
        ['KNN', 'Naive Bayes', 'Decision Tree'],
        accuracies,
        save_path=f"{Config.RESULTS_DIR}/confusion_matrices_comparison.png"
    )
    
    # Plot accuracy comparison bar chart
    plot_model_comparison_bar(
        ['KNN', 'Naive Bayes', 'Decision Tree'],
        accuracies,
        save_path=f"{Config.RESULTS_DIR}/model_accuracy_comparison.png"
    )
    
    # Save models
    for name, model in training_results['models'].items():
        filename = name.lower().replace(' ', '_')
        save_model(model, f"{Config.MODELS_DIR}/{filename}_model.pkl")
    
    # Export comparison results
    comparison_df.to_csv(
        f"{Config.RESULTS_DIR}/model_comparison_results.csv",
        index=False
    )
    print(f"✓ Model comparison saved to {Config.RESULTS_DIR}/model_comparison_results.csv")
    
    print("\n✓ Phase 4 Complete")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline():
    """
    Execute the complete machine learning pipeline.
    """
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("LOAN DEFAULT PREDICTION - COMPLETE ML PIPELINE")
    print("="*80)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Author: Swithun M. Chiziko")
    print(f"Institution: Curtin University")
    print("="*80)
    
    # Setup
    setup_directories()
    
    try:
        # Phase 1: Data Preprocessing
        data_splits = phase1_data_preprocessing()
        
        # Phase 2: Feature Engineering
        engineered_data = phase2_feature_engineering(data_splits)
        
        # Phase 3: Model Training
        training_results = phase3_model_training(engineered_data)
        
        # Phase 4: Comparison and Export
        phase4_comparison_and_export(training_results, engineered_data)
        
        # Success
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*80)
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"\nResults saved to: {Config.RESULTS_DIR}/")
        print(f"Models saved to: {Config.MODELS_DIR}/")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    success = run_complete_pipeline()
    
    if success:
        print("\n✅ All operations completed successfully!")
    else:
        print("\n❌ Pipeline encountered errors. Please check the logs above.")
