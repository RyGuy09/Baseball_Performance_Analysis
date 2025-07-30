"""
Comprehensive model evaluation framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from datetime import datetime
import json

from .metrics import ClassificationMetrics, RegressionMetrics
from ..utils.helpers import save_json, ensure_dir, timer
from ..utils.constants import PERFORMANCE_BENCHMARKS

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Base class for model evaluation"""
    
    def __init__(self, model: Any, model_name: str,
                 save_dir: Optional[Path] = None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            model_name: Name of the model
            save_dir: Directory to save evaluation results
        """
        self.model = model
        self.model_name = model_name
        self.save_dir = Path(save_dir) if save_dir else Path('evaluation_results')
        ensure_dir(self.save_dir)
        
        self.results = {}
        self.plots = {}
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform complete model evaluation
        
        Args:
            X_test: Test features
            y_test: Test targets
            feature_names: Feature names
            
        Returns:
            Dictionary of evaluation results
        """
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def save_results(self, include_plots: bool = True):
        """Save evaluation results and plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results JSON
        results_path = self.save_dir / f"{self.model_name}_evaluation_{timestamp}.json"
        save_json(self.results, results_path)
        logger.info(f"Results saved to {results_path}")
        
        # Save plots
        if include_plots and self.plots:
            plots_dir = self.save_dir / 'plots' / self.model_name
            ensure_dir(plots_dir)
            
            for plot_name, fig in self.plots.items():
                plot_path = plots_dir / f"{plot_name}_{timestamp}.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {plot_path}")
                plt.close(fig)
    
    def generate_report(self) -> str:
        """Generate text evaluation report"""
        report_lines = [
            f"Model Evaluation Report: {self.model_name}",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Add metrics
        if 'metrics' in self.results:
            report_lines.append("Performance Metrics:")
            report_lines.append("-" * 30)
            for metric, value in self.results['metrics'].items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric}: {value}")
            report_lines.append("")
        
        # Add benchmark comparison
        if 'benchmark_comparison' in self.results:
            report_lines.append("Benchmark Comparison:")
            report_lines.append("-" * 30)
            for level, status in self.results['benchmark_comparison'].items():
                report_lines.append(f"  {level}: {'✓' if status else '✗'}")
            report_lines.append("")
        
        return "\n".join(report_lines)


class ClassificationEvaluator(ModelEvaluator):
    """Evaluator for classification models"""
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate classification model"""
        logger.info(f"Evaluating classification model: {self.model_name}")
        
        with timer("Model evaluation"):
            # Get predictions
            y_pred = self.model.predict(X_test)
            y_proba = None
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_test)
            
            # Calculate metrics
            metrics_obj = ClassificationMetrics(y_test, y_pred, y_proba)
            self.results['metrics'] = metrics_obj.metrics
            
            # Get curve data
            self.results['curves'] = metrics_obj.get_curve_data()
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_') and feature_names:
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.results['feature_importance'] = importance_df.to_dict('records')
            
            # Benchmark comparison
            self.results['benchmark_comparison'] = self._compare_to_benchmarks()
            
            # Generate plots
            self._generate_plots(y_test, y_pred, y_proba)
            
            # Additional analysis
            self.results['prediction_analysis'] = self._analyze_predictions(y_test, y_pred, y_proba)
        
        return self.results
    
    def _compare_to_benchmarks(self) -> Dict[str, bool]:
        """Compare model performance to benchmarks"""
        benchmarks = PERFORMANCE_BENCHMARKS['classification']
        comparison = {}
        
        metrics = self.results['metrics']
        
        for level, thresholds in benchmarks.items():
            passed = all(
                metrics.get(metric, 0) >= threshold
                for metric, threshold in thresholds.items()
                if metric in metrics
            )
            comparison[level] = passed
        
        return comparison
    
    def _generate_plots(self, y_test: np.ndarray, y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None):
        """Generate evaluation plots"""
        # Confusion matrix
        self.plots['confusion_matrix'] = self._plot_confusion_matrix(y_test, y_pred)
        
        # ROC and PR curves
        if y_proba is not None and len(np.unique(y_test)) == 2:
            self.plots['roc_curve'] = self._plot_roc_curve(y_test, y_proba[:, 1])
            self.plots['pr_curve'] = self._plot_pr_curve(y_test, y_proba[:, 1])
            self.plots['calibration'] = self._plot_calibration_curve(y_test, y_proba[:, 1])
        
        # Feature importance
        if 'feature_importance' in self.results:
            self.plots['feature_importance'] = self._plot_feature_importance()
    
    def _plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {self.model_name}')
        
        return fig
    
    def _plot_roc_curve(self, y_test: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {self.model_name}')
        ax.legend(loc="lower right")
        
        return fig
    
    def _plot_pr_curve(self, y_test: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
        """Plot Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        ax.step(recall, precision, where='post', color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline
        baseline = y_test.mean()
        ax.axhline(y=baseline, color='navy', lw=2, linestyle='--',
                   label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {self.model_name}')
        ax.legend(loc="lower left")
        
        return fig
    
    def _plot_calibration_curve(self, y_test: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
        """Plot calibration curve"""
        from sklearn.calibration import calibration_curve
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_proba, n_bins=10, strategy='uniform'
        )
        
        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Plot model calibration
        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                color='darkorange', label=self.model_name)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Plot')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_feature_importance(self, top_n: int = 20) -> plt.Figure:
        """Plot feature importance"""
        importance_data = pd.DataFrame(self.results['feature_importance'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top N features
        top_features = importance_data.head(top_n)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], color='skyblue')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances - {self.model_name}')
        
        # Add value labels
        for i, v in enumerate(top_features['importance']):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        return fig
    
    def _analyze_predictions(self, y_test: np.ndarray, y_pred: np.ndarray,
                           y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze prediction patterns"""
        analysis = {}
        
        # Prediction distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        analysis['prediction_distribution'] = {
            int(k): int(v) for k, v in zip(unique, counts)
        }
        
        # Error analysis
        errors = y_test != y_pred
        analysis['error_rate'] = float(errors.mean())
        
        if y_proba is not None and len(np.unique(y_test)) == 2:
            # Confidence analysis
            proba_positive = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            
            # Analyze correct predictions
            correct_mask = y_test == y_pred
            analysis['confidence_correct'] = {
                'mean': float(proba_positive[correct_mask].mean()),
                'std': float(proba_positive[correct_mask].std()),
                'min': float(proba_positive[correct_mask].min()),
                'max': float(proba_positive[correct_mask].max())
            }
            
            # Analyze incorrect predictions
            analysis['confidence_incorrect'] = {
                'mean': float(proba_positive[~correct_mask].mean()),
                'std': float(proba_positive[~correct_mask].std()),
                'min': float(proba_positive[~correct_mask].min()),
                'max': float(proba_positive[~correct_mask].max())
            }
            
            # High confidence errors
            high_conf_errors = (proba_positive > 0.8) & (y_test == 0) | \
                              (proba_positive < 0.2) & (y_test == 1)
            analysis['high_confidence_error_rate'] = float(high_conf_errors.mean())
        
        return analysis


class RegressionEvaluator(ModelEvaluator):
    """Evaluator for regression models"""
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate regression model"""
        logger.info(f"Evaluating regression model: {self.model_name}")
        
        with timer("Model evaluation"):
            # Get predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics_obj = RegressionMetrics(y_test, y_pred)
            self.results['metrics'] = metrics_obj.metrics
            
            # Residual analysis
            self.results['residual_analysis'] = metrics_obj.get_residual_analysis()
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_') and feature_names:
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.results['feature_importance'] = importance_df.to_dict('records')
            
            # Benchmark comparison
            self.results['benchmark_comparison'] = self._compare_to_benchmarks()
            
            # Generate plots
            self._generate_plots(y_test, y_pred)
            
            # Additional analysis
            self.results['prediction_analysis'] = self._analyze_predictions(y_test, y_pred)
        
        return self.results
    
    def _compare_to_benchmarks(self) -> Dict[str, bool]:
        """Compare model performance to benchmarks"""
        benchmarks = PERFORMANCE_BENCHMARKS['regression']
        comparison = {}
        
        metrics = self.results['metrics']
        
        for level, thresholds in benchmarks.items():
            passed = True
            for metric, threshold in thresholds.items():
                if metric in metrics:
                    if metric in ['rmse', 'mae']:  # Lower is better
                        passed = passed and (metrics[metric] <= threshold)
                    else:  # Higher is better
                        passed = passed and (metrics[metric] >= threshold)
            comparison[level] = passed
        
        return comparison
    
    def _generate_plots(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Generate evaluation plots"""
        self.plots['predicted_vs_actual'] = self._plot_predicted_vs_actual(y_test, y_pred)
        self.plots['residual_plot'] = self._plot_residuals(y_test, y_pred)
        self.plots['residual_distribution'] = self._plot_residual_distribution(y_test, y_pred)
        self.plots['qq_plot'] = self._plot_qq(y_test, y_pred)
        
        if 'feature_importance' in self.results:
            self.plots['feature_importance'] = self._plot_feature_importance()
    
    def _plot_predicted_vs_actual(self, y_test: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """Plot predicted vs actual values"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
                label='Perfect prediction')
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Predicted vs Actual - {self.model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_residuals(self, y_test: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """Plot residuals vs predicted values"""
        residuals = y_test - y_pred
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        # Add confidence bands
        std_resid = np.std(residuals)
        ax.axhline(y=2*std_resid, color='r', linestyle=':', alpha=0.5)
        ax.axhline(y=-2*std_resid, color='r', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residual Plot - {self.model_name}')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_residual_distribution(self, y_test: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """Plot residual distribution"""
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7, density=True)
        
        # Add normal distribution overlay
        from scipy import stats
        mu, std = stats.norm.fit(residuals)
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax1.plot(x, p, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
        
        ax1.set_xlabel('Residuals')
        ax1.set_ylabel('Density')
        ax1.set_title('Residual Distribution')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(residuals, vert=True)
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {self.model_name}')
        plt.tight_layout()
        
        return fig
    
    def _plot_qq(self, y_test: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """Plot Q-Q plot for residuals"""
        from scipy import stats
        
        residuals = y_test - y_pred
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot - {self.model_name}')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_feature_importance(self, top_n: int = 20) -> plt.Figure:
        """Plot feature importance"""
        # Same as classification version
        importance_data = pd.DataFrame(self.results['feature_importance'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_data.head(top_n)
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], color='lightgreen')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances - {self.model_name}')
        
        for i, v in enumerate(top_features['importance']):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        return fig
    
    def _analyze_predictions(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction patterns"""
        residuals = y_test - y_pred
        
        analysis = {
            'prediction_range': {
                'min': float(y_pred.min()),
                'max': float(y_pred.max()),
                'mean': float(y_pred.mean()),
                'std': float(y_pred.std())
            },
            'largest_errors': {
                'overestimation': {
                    'value': float(residuals.min()),
                    'index': int(residuals.argmin())
                },
                'underestimation': {
                    'value': float(residuals.max()),
                    'index': int(residuals.argmax())
                }
            }
        }
        
        # Systematic bias check
        if len(y_test) >= 10:
            # Split into bins and check bias
            n_bins = min(10, len(y_test) // 10)
            bins = pd.qcut(y_test, n_bins, labels=False, duplicates='drop')
            
            bin_bias = []
            for bin_idx in range(n_bins):
                mask = bins == bin_idx
                if mask.sum() > 0:
                    bin_residual = residuals[mask].mean()
                    bin_bias.append({
                        'bin': int(bin_idx),
                        'actual_mean': float(y_test[mask].mean()),
                        'predicted_mean': float(y_pred[mask].mean()),
                        'mean_residual': float(bin_residual)
                    })
            
            analysis['bin_bias'] = bin_bias
        
        return analysis


class EnsembleEvaluator(ModelEvaluator):
    """Evaluator for ensemble models"""
    
    def __init__(self, models: Dict[str, Any], ensemble_name: str,
                 save_dir: Optional[Path] = None):
        """
        Initialize ensemble evaluator
        
        Args:
            models: Dictionary of model_name -> model
            ensemble_name: Name of the ensemble
            save_dir: Directory to save results
        """
        super().__init__(None, ensemble_name, save_dir)
        self.models = models
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                feature_names: Optional[List[str]] = None,
                task_type: str = 'classification') -> Dict[str, Any]:
        """Evaluate ensemble and component models"""
        logger.info(f"Evaluating ensemble: {self.model_name}")
        
        # Evaluate each model
        individual_results = {}
        
        for model_name, model in self.models.items():
            if task_type == 'classification':
                evaluator = ClassificationEvaluator(model, model_name, self.save_dir)
            else:
                evaluator = RegressionEvaluator(model, model_name, self.save_dir)
            
            individual_results[model_name] = evaluator.evaluate(X_test, y_test, feature_names)
        
        # Store individual results
        self.results['individual_models'] = individual_results
        
        # Compare models
        self.results['model_comparison'] = self._compare_models(individual_results, task_type)
        
        # Generate comparison plots
        self._generate_comparison_plots(individual_results, task_type)
        
        return self.results
    
    def _compare_models(self, results: Dict[str, Dict], task_type: str) -> pd.DataFrame:
        """Compare model performances"""
        comparison_data = []
        
        for model_name, model_results in results.items():
            metrics = model_results.get('metrics', {})
            
            row = {'model': model_name}
            
            if task_type == 'classification':
                key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            else:
                key_metrics = ['rmse', 'mae', 'r2', 'mape']
            
            for metric in key_metrics:
                if metric in metrics:
                    row[metric] = metrics[metric]
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _generate_comparison_plots(self, results: Dict[str, Dict], task_type: str):
        """Generate comparison plots"""
        comparison_df = self.results['model_comparison']
        
        # Metrics comparison
        self.plots['metrics_comparison'] = self._plot_metrics_comparison(comparison_df, task_type)
        
        # ROC curves comparison (classification only)
        if task_type == 'classification':
            self.plots['roc_comparison'] = self._plot_roc_comparison(results)
    
    def _plot_metrics_comparison(self, df: pd.DataFrame, task_type: str) -> plt.Figure:
        """Plot metrics comparison across models"""
        if task_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        else:
            metrics = ['rmse', 'mae', 'r2', 'mape']
        
        # Filter available metrics
        metrics = [m for m in metrics if m in df.columns]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        x = np.arange(len(df))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2) * width
            ax.bar(x + offset, df[metric], width, label=metric)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def _plot_roc_comparison(self, results: Dict[str, Dict]) -> plt.Figure:
        """Plot ROC curves comparison"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for model_name, model_results in results.items():
            curves = model_results.get('curves', {})
            if 'roc_curve' in curves:
                roc_data = curves['roc_curve']
                ax.plot(roc_data['fpr'], roc_data['tpr'],
                       label=f"{model_name} (AUC = {roc_data['auc']:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        return fig


# Convenience function
def create_evaluation_report(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                           model_name: str, task_type: str = 'classification',
                           feature_names: Optional[List[str]] = None,
                           save_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Create comprehensive evaluation report for a model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        task_type: 'classification' or 'regression'
        feature_names: Feature names
        save_dir: Directory to save results
        
    Returns:
        Evaluation results dictionary
    """
    if task_type == 'classification':
        evaluator = ClassificationEvaluator(model, model_name, save_dir)
    else:
        evaluator = RegressionEvaluator(model, model_name, save_dir)
    
    results = evaluator.evaluate(X_test, y_test, feature_names)
    evaluator.save_results()
    
    # Print summary
    report = evaluator.generate_report()
    print(report)
    
    return results


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    results = create_evaluation_report(
        model, X_test, y_test,
        model_name="RandomForest",
        task_type="classification",
        feature_names=[f"feature_{i}" for i in range(20)]
    )
    
    print(f"\nTest Accuracy: {results['metrics']['accuracy']:.3f}")