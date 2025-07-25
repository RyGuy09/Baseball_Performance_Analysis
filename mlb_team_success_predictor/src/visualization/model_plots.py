"""
Model performance and evaluation visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
from datetime import datetime

from ..utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Create model evaluation and performance plots"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6),
                 style: str = 'whitegrid',
                 save_dir: Optional[Path] = None):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
            style: Seaborn style
            save_dir: Directory to save plots
        """
        self.figsize = figsize
        self.style = style
        self.save_dir = Path(save_dir) if save_dir else Path('plots/models')
        ensure_dir(self.save_dir)
        
        sns.set_style(style)
    
    def plot_confusion_matrix_advanced(self, y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     labels: Optional[List[str]] = None,
                                     normalize: bool = True,
                                     save: bool = True,
                                     model_name: str = 'Model') -> plt.Figure:
        """
        Plot advanced confusion matrix with percentages
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize
            save: Whether to save the plot
            model_name: Name of the model
            
        Returns:
            Figure object
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm_normalized = cm
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2%' if normalize else 'd',
                   cmap='Blues', cbar=True, ax=ax1,
                   xticklabels=labels, yticklabels=labels)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_title(f'Confusion Matrix - {model_name}' + 
                     (' (Normalized)' if normalize else ''))
        
        # Plot 2: Count-based confusion matrix with percentages
        # Create text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = cm_normalized[i, j] * 100 if normalize else (count / cm.sum()) * 100
                
                ax2.text(j, i, f'{count}\n({percentage:.1f}%)',
                        ha="center", va="center",
                        color="white" if count > thresh else "black")
        
        # Create heatmap without annotations
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True, ax=ax2,
                   xticklabels=labels, yticklabels=labels)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title(f'Confusion Matrix with Counts - {model_name}')
        
        plt.suptitle(f'{model_name} Classification Results', fontsize=14)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{model_name.lower()}_confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_roc_curves_comparison(self, model_results: Dict[str, Dict],
                                  save: bool = True) -> plt.Figure:
        """
        Plot ROC curves for multiple models
        
        Args:
            model_results: Dict of model_name -> {'y_true', 'y_proba'}
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot ROC curve for each model
        for model_name, results in model_results.items():
            y_true = results['y_true']
            y_proba = results['y_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot random classifier
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Add diagonal reference areas
        ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
        
        if save:
            save_path = self.save_dir / 'roc_curves_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pr_curves_comparison(self, model_results: Dict[str, Dict],
                                 save: bool = True) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models
        
        Args:
            model_results: Dict of model_name -> {'y_true', 'y_proba'}
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Calculate baseline (random classifier)
        baseline = np.mean([results['y_true'].mean() 
                          for results in model_results.values()])
        
        # Plot PR curve for each model
        for model_name, results in model_results.items():
            y_true = results['y_true']
            y_proba = results['y_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            
            # Calculate average precision
            from sklearn.metrics import average_precision_score
            avg_precision = average_precision_score(y_true, y_proba)
            
            ax.plot(recall, precision, linewidth=2,
                   label=f'{model_name} (AP = {avg_precision:.3f})')
        
        # Plot baseline
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=2,
                  label=f'Baseline (AP = {baseline:.3f})')
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        if save:
            save_path = self.save_dir / 'pr_curves_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance_comparison(self, importance_data: Dict[str, pd.DataFrame],
                                         top_n: int = 20,
                                         save: bool = True) -> plt.Figure:
        """
        Compare feature importance across models
        
        Args:
            importance_data: Dict of model_name -> importance DataFrame
            top_n: Number of top features to show
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        # Get all unique features
        all_features = set()
        for df in importance_data.values():
            all_features.update(df['feature'].tolist()[:top_n])
        
        # Create comparison dataframe
        comparison_data = []
        for feature in all_features:
            row = {'feature': feature}
            for model_name, df in importance_data.items():
                feature_row = df[df['feature'] == feature]
                if not feature_row.empty:
                    row[model_name] = feature_row['importance'].values[0]
                else:
                    row[model_name] = 0
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate average importance and sort
        model_names = list(importance_data.keys())
        comparison_df['avg_importance'] = comparison_df[model_names].mean(axis=1)
        comparison_df = comparison_df.sort_values('avg_importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot grouped bar chart
        x = np.arange(len(comparison_df))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            offset = (i - len(model_names)/2) * width + width/2
            ax.bar(x + offset, comparison_df[model_name], width,
                  label=model_name, alpha=0.8)
        
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['feature'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'feature_importance_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_regression_diagnostics(self, y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  model_name: str = 'Model',
                                  save: bool = True) -> plt.Figure:
        """
        Create comprehensive regression diagnostic plots
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Predicted vs Actual
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Predicted vs Actual')
        
        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # 2. Residuals vs Predicted
        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        
        # Add reference lines at ±2 std
        std_resid = np.std(residuals)
        ax.axhline(y=2*std_resid, color='r', linestyle=':', alpha=0.5)
        ax.axhline(y=-2*std_resid, color='r', linestyle=':', alpha=0.5)
        
        # 3. Residual Distribution
        ax = axes[0, 2]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, density=True)
        
        # Add normal distribution overlay
        from scipy import stats
        mu, std = stats.norm.fit(residuals)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'r-', linewidth=2, label=f'N({mu:.1f}, {std:.1f})')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.legend()
        
        # 4. Q-Q Plot
        ax = axes[1, 0]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # 5. Scale-Location Plot
        ax = axes[1, 1]
        standardized_residuals = residuals / np.sqrt(np.abs(residuals))
        ax.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('√|Standardized Residuals|')
        ax.set_title('Scale-Location Plot')
        
        # Add smoothed line
        from scipy.ndimage import gaussian_filter1d
        sorted_idx = np.argsort(y_pred)
        smoothed = gaussian_filter1d(np.sqrt(np.abs(standardized_residuals[sorted_idx])), 5)
        ax.plot(y_pred[sorted_idx], smoothed, 'r-', linewidth=2)
        
        # 6. Residuals vs Leverage (simplified)
        ax = axes[1, 2]
        # Calculate Cook's distance (simplified)
        n = len(residuals)
        p = 1  # Simplified - should be number of parameters
        leverage = 1/n  # Simplified - should calculate actual leverage
        cooks_d = (residuals**2 / (p * std_resid**2)) * (leverage / (1 - leverage)**2)
        
        ax.scatter(range(len(residuals)), cooks_d, alpha=0.5)
        ax.set_xlabel('Index')
        ax.set_ylabel("Cook's Distance")
        ax.set_title("Cook's Distance Plot")
        
        # Add reference line
        ax.axhline(y=4/n, color='r', linestyle='--', label='4/n threshold')
        ax.legend()
        
        plt.suptitle(f'Regression Diagnostics - {model_name}', fontsize=14)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{model_name.lower()}_regression_diagnostics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_intervals(self, y_true: np.ndarray,
                                y_pred: np.ndarray,
                                lower_bound: np.ndarray,
                                upper_bound: np.ndarray,
                                sample_indices: Optional[np.ndarray] = None,
                                model_name: str = 'Model',
                                save: bool = True) -> plt.Figure:
        """
        Plot predictions with confidence/prediction intervals
        
        Args:
            y_true: True values
            y_pred: Predicted values
            lower_bound: Lower bound of intervals
            upper_bound: Upper bound of intervals
            sample_indices: Indices for x-axis (optional)
            model_name: Name of the model
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        if sample_indices is None:
            sample_indices = np.arange(len(y_true))
        
        # Sort by predicted values for better visualization
        sort_idx = np.argsort(y_pred)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot prediction intervals
        ax.fill_between(sample_indices, 
                       lower_bound[sort_idx], 
                       upper_bound[sort_idx],
                       alpha=0.3, color='gray', label='95% Prediction Interval')
        
        # Plot predictions
        ax.plot(sample_indices, y_pred[sort_idx], 'b-', linewidth=2, label='Predictions')
        
        # Plot actual values
        ax.scatter(sample_indices, y_true[sort_idx], color='red', alpha=0.6, 
                  s=20, label='Actual')
        
        # Calculate coverage
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        
        ax.set_xlabel('Sample Index (sorted by prediction)')
        ax.set_ylabel('Value')
        ax.set_title(f'Prediction Intervals - {model_name}\nCoverage: {coverage:.1%}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            save_path = self.save_dir / f'{model_name.lower()}_prediction_intervals.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_learning_curves(self, train_sizes: np.ndarray,
                           train_scores: np.ndarray,
                           val_scores: np.ndarray,
                           model_name: str = 'Model',
                           metric_name: str = 'Score',
                           save: bool = True) -> plt.Figure:
        """
        Plot learning curves
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores (n_sizes, n_cv_folds)
            val_scores: Validation scores (n_sizes, n_cv_folds)
            model_name: Name of the model
            metric_name: Name of the metric
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        ax.plot(train_sizes, train_mean, 'o-', color='blue', linewidth=2,
               markersize=8, label='Training score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.1, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', linewidth=2,
               markersize=8, label='Cross-validation score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.1, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Learning Curves - {model_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add convergence information
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        gap = final_train - final_val
        
        info_text = f'Final Training: {final_train:.3f}\n'
        info_text += f'Final Validation: {final_val:.3f}\n'
        info_text += f'Gap: {gap:.3f}'
        
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save:
            save_path = self.save_dir / f'{model_name.lower()}_learning_curves.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_model_comparison_report(self, model_results: Dict[str, Dict],
                                     output_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive model comparison report
        
        Args:
            model_results: Results for each model
            output_path: Path to save the report
            
        Returns:
            Figure object
        """
        n_models = len(model_results)
        fig = plt.figure(figsize=(16, 4 * n_models))
        
        # Create grid
        gs = fig.add_gridspec(n_models, 4, hspace=0.3, wspace=0.3)
        
        for i, (model_name, results) in enumerate(model_results.items()):
            # Confusion matrix
            if 'confusion_matrix' in results:
                ax = fig.add_subplot(gs[i, 0])
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'{model_name}\nConfusion Matrix')
            
            # Feature importance
            if 'feature_importance' in results:
                ax = fig.add_subplot(gs[i, 1])
                imp_df = results['feature_importance'][:10]
                ax.barh(imp_df['feature'], imp_df['importance'])
                ax.set_xlabel('Importance')
                ax.set_title(f'{model_name}\nTop 10 Features')
            
            # Metrics
            if 'metrics' in results:
                ax = fig.add_subplot(gs[i, 2])
                ax.axis('off')
                metrics_text = f"{model_name} Metrics:\n\n"
                for metric, value in results['metrics'].items():
                    if isinstance(value, float):
                        metrics_text += f"{metric}: {value:.3f}\n"
                ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='center')
            
            # ROC/PR curves
            if 'curves' in results:
                ax = fig.add_subplot(gs[i, 3])
                if 'roc' in results['curves']:
                    fpr, tpr = results['curves']['roc']['fpr'], results['curves']['roc']['tpr']
                    auc_score = results['curves']['roc']['auc']
                    ax.plot(fpr, tpr, label=f'ROC (AUC={auc_score:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    ax.set_xlabel('FPR')
                    ax.set_ylabel('TPR')
                    ax.set_title(f'{model_name}\nROC Curve')
                    ax.legend()
        
        plt.suptitle('Model Comparison Report', fontsize=16)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig


# Convenience functions
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Optional[List[str]] = None,
                         model_name: str = 'Model',
                         save_dir: Optional[Path] = None) -> plt.Figure:
    """Quick function to plot confusion matrix"""
    visualizer = ModelVisualizer(save_dir=save_dir)
    return visualizer.plot_confusion_matrix_advanced(y_true, y_pred, labels, 
                                                   model_name=model_name)


def plot_roc_curves(model_results: Dict[str, Dict],
                   save_dir: Optional[Path] = None) -> plt.Figure:
    """Quick function to plot ROC curves comparison"""
    visualizer = ModelVisualizer(save_dir=save_dir)
    return visualizer.plot_roc_curves_comparison(model_results)


def plot_feature_importance(importance_df: pd.DataFrame,
                          top_n: int = 20,
                          model_name: str = 'Model',
                          save_dir: Optional[Path] = None) -> plt.Figure:
    """Quick function to plot feature importance"""
    visualizer = ModelVisualizer(save_dir=save_dir)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'], color='skyblue')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances - {model_name}')
    
    for i, v in enumerate(top_features['importance']):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_dir:
        ensure_dir(save_dir)
        save_path = save_dir / f'{model_name.lower()}_feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prediction_results(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = 'Model',
                          task_type: str = 'regression',
                          save_dir: Optional[Path] = None) -> plt.Figure:
    """Quick function to plot prediction results"""
    visualizer = ModelVisualizer(save_dir=save_dir)
    
    if task_type == 'regression':
        return visualizer.plot_regression_diagnostics(y_true, y_pred, model_name)
    else:
        # For classification, create simple prediction distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        x = np.arange(len(unique_true))
        width = 0.35
        
        ax.bar(x - width/2, counts_true, width, label='Actual', alpha=0.8)
        ax.bar(x + width/2, counts_pred, width, label='Predicted', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(f'Prediction Distribution - {model_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_true)
        ax.legend()
        
        return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                  model_name: str = 'Model',
                  save_dir: Optional[Path] = None) -> plt.Figure:
    """Quick function to plot residuals"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residual plot
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'Residual Plot - {model_name}')
    ax1.grid(True, alpha=0.3)
    
    # Residual histogram
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        ensure_dir(save_dir)
        save_path = save_dir / f'{model_name.lower()}_residuals.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_model_report(model_results: Dict[str, Any],
                       output_dir: Path,
                       model_name: str = 'Model') -> Dict[str, Any]:
    """
    Create comprehensive model evaluation report
    
    Args:
        model_results: Model evaluation results
        output_dir: Directory for output
        model_name: Name of the model
        
    Returns:
        Summary dictionary
    """
    ensure_dir(output_dir)
    visualizer = ModelVisualizer(save_dir=output_dir)
    
    plots_created = []
    
    # Create appropriate plots based on available data
    if 'confusion_matrix' in model_results and 'y_true' in model_results:
        fig = visualizer.plot_confusion_matrix_advanced(
            model_results['y_true'],
            model_results['y_pred'],
            model_name=model_name
        )
        plots_created.append('confusion_matrix')
        plt.close(fig)
    
    if 'y_true' in model_results and 'y_pred' in model_results:
        if 'task_type' in model_results and model_results['task_type'] == 'regression':
            fig = visualizer.plot_regression_diagnostics(
                model_results['y_true'],
                model_results['y_pred'],
                model_name=model_name
            )
            plots_created.append('regression_diagnostics')
            plt.close(fig)
    
    if 'feature_importance' in model_results:
        fig = plot_feature_importance(
            pd.DataFrame(model_results['feature_importance']),
            model_name=model_name,
            save_dir=output_dir
        )
        plots_created.append('feature_importance')
        plt.close(fig)
    
    summary = {
        'model_name': model_name,
        'plots_created': plots_created,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save summary
    import json
    with open(output_dir / f'{model_name.lower()}_report_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Model report created in {output_dir}")
    
    return summary