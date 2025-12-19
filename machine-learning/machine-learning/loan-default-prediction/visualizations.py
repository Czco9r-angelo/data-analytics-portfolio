"""
Visualization Module for Loan Default Prediction
================================================

This module generates all visualizations for:
- Exploratory Data Analysis (EDA)
- Distribution analysis
- Correlation heatmaps
- Model performance comparisons
- Confusion matrices

Author: Swithun M. Chiziko
Date: June 2022
Last Edited: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.gridspec import GridSpec


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# DIAGNOSTIC PLOTS FOR NORMALITY TESTING
# ============================================================================

def plot_diagnostic_normality(df, column, save_path=None):
    """
    Create diagnostic plots to test normality of a variable.
    
    Includes:
    - Histogram
    - Q-Q plot
    - CDF vs Normal CDF
    - Box plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to analyze
    save_path : str, optional
        Path to save the figure
    """
    # Remove NaN values
    data = df[column].dropna()
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Calculate statistics
    mean = data.mean()
    median = data.median()
    std = data.std()
    skew = data.skew()
    
    # 1. Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
    ax1.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.3f}')
    ax1.set_xlabel(column, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title(f'Histogram', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Q-Q Plot
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title('Probability Plot', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. CDF vs Normal CDF
    ax3 = fig.add_subplot(gs[1, 0])
    # Sort data
    sorted_data = np.sort(data)
    # Empirical CDF
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax3.plot(sorted_data, ecdf, linewidth=2, color='steelblue', label='Empirical CDF')
    # Theoretical normal CDF
    theoretical_cdf = stats.norm.cdf(sorted_data, mean, std)
    ax3.plot(sorted_data, theoretical_cdf, linewidth=2, 
             color='red', linestyle='--', label='Normal CDF')
    ax3.set_xlabel(column, fontsize=11, fontweight='bold')
    ax3.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax3.set_title('CDF Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Box Plot
    ax4 = fig.add_subplot(gs[1, 1])
    bp = ax4.boxplot(data, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='black'),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'))
    ax4.set_ylabel(column, fontsize=11, fontweight='bold')
    ax4.set_title('Box Plot (Outlier Detection)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add overall title with statistics
    fig.suptitle(
        f'{column} - Normality Diagnostic Plots\n' +
        f'Mean: {mean:.3f} | Median: {median:.3f} | Std: {std:.3f} | Skew: {skew:.3f}',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Diagnostic plot for {column} saved to {save_path}")
    
    plt.show()
    
    # Interpretation
    print(f"\n{column} Distribution Analysis:")
    print("=" * 50)
    print(f"Mean: {mean:.6f}")
    print(f"Median: {median:.6f}")
    print(f"Standard Deviation: {std:.6f}")
    print(f"Skewness: {skew:.6f}")
    
    if abs(skew) < 0.5:
        skew_interpretation = "approximately symmetric"
    elif skew > 0:
        skew_interpretation = "positively skewed (right tail)"
    else:
        skew_interpretation = "negatively skewed (left tail)"
    
    print(f"Interpretation: Distribution is {skew_interpretation}")
    
    # Outlier count
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
    print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")


# ============================================================================
# CORRELATION HEATMAP
# ============================================================================

def plot_correlation_heatmap(df, figsize=(20, 16), save_path=None):
    """
    Create comprehensive correlation heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    
    plt.title('Correlation Matrix Heatmap', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved to {save_path}")
    
    plt.show()
    
    # Print highly correlated pairs
    print("\nHighly Correlated Variable Pairs (|r| > 0.9):")
    print("=" * 60)
    
    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find high correlations
    high_corr = []
    for column in upper_tri.columns:
        for index in upper_tri.index:
            corr_value = upper_tri.loc[index, column]
            if abs(corr_value) > 0.9:
                high_corr.append((index, column, corr_value))
    
    # Sort by absolute correlation
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for var1, var2, corr in high_corr:
        print(f"{var1:20} ↔ {var2:20} | r = {corr:7.4f}")


# ============================================================================
# CONFUSION MATRIX COMPARISON
# ============================================================================

def plot_confusion_matrices_comparison(confusion_matrices, model_names, 
                                      accuracies, save_path=None):
    """
    Plot multiple confusion matrices side-by-side for comparison.
    
    Parameters:
    -----------
    confusion_matrices : list
        List of confusion matrices
    model_names : list
        List of model names
    accuracies : list
        List of accuracy scores
    save_path : str, optional
        Path to save the figure
    """
    n_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (cm, name, acc) in enumerate(zip(confusion_matrices, model_names, accuracies)):
        ax = axes[idx]
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Count'},
            linewidths=1,
            linecolor='black'
        )
        
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}\nAccuracy: {acc:.2%}', 
                    fontsize=12, fontweight='bold')
    
    plt.suptitle('Confusion Matrix Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix comparison saved to {save_path}")
    
    plt.show()


# ============================================================================
# BATCH DIAGNOSTIC PLOTS
# ============================================================================

def generate_all_diagnostic_plots(df, numerical_columns, output_dir='results/diagnostics'):
    """
    Generate diagnostic plots for all numerical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_columns : list
        List of numerical column names
    output_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating diagnostic plots for {len(numerical_columns)} variables...")
    print("=" * 70)
    
    for col in numerical_columns:
        print(f"\nProcessing: {col}")
        save_path = f"{output_dir}/{col}_diagnostics.png"
        
        try:
            plot_diagnostic_normality(df, col, save_path=save_path)
        except Exception as e:
            print(f"  ⚠️  Error generating plot for {col}: {str(e)}")
    
    print("\n" + "=" * 70)
    print(f"✓ All diagnostic plots saved to {output_dir}/")


# ============================================================================
# DISTRIBUTION PLOTS
# ============================================================================

def plot_distributions_grid(df, columns, rows=4, cols=4, figsize=(16, 12),
                           save_path=None):
    """
    Create grid of distribution plots for multiple variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of column names to plot
    rows, cols : int
        Grid dimensions
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(columns[:rows*cols]):
        ax = axes[idx]
        data = df[col].dropna()
        
        # Histogram
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Add mean line
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label=f'μ={data.mean():.2f}')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(columns), rows*cols):
        axes[idx].axis('off')
    
    plt.suptitle('Variable Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution grid saved to {save_path}")
    
    plt.show()


# ============================================================================
# MODEL PERFORMANCE VISUALIZATION
# ============================================================================

def plot_model_comparison_bar(model_names, accuracies, save_path=None):
    """
    Create bar chart comparing model accuracies.
    
    Parameters:
    -----------
    model_names : list
        List of model names
    accuracies : list
        List of accuracy scores
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(model_names)]
    bars = plt.bar(model_names, accuracies, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison chart saved to {save_path}")
    
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("\nAvailable visualization functions:")
    print("  - plot_diagnostic_normality()")
    print("  - plot_correlation_heatmap()")
    print("  - plot_confusion_matrices_comparison()")
    print("  - generate_all_diagnostic_plots()")
    print("  - plot_distributions_grid()")
    print("  - plot_model_comparison_bar()")
