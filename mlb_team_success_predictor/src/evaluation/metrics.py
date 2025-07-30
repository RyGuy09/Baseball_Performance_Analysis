"""
Comprehensive metrics for model evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
    
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score,
    
    # Other
    make_scorer
)
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Calculate and store classification metrics"""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None,
                 labels: Optional[List[str]] = None,
                 pos_label: int = 1):
        """
        Initialize classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            labels: Class labels
            pos_label: Positive class label for binary classification
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.labels = labels
        self.pos_label = pos_label
        
        # Determine if binary or multiclass
        self.n_classes = len(np.unique(y_true))
        self.is_binary = self.n_classes == 2
        
        # Calculate all metrics
        self.metrics = self._calculate_all_metrics()
        
    def _calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        
        # Precision, recall, F1
        average = 'binary' if self.is_binary else 'weighted'
        metrics['precision'] = precision_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )
        metrics['recall'] = recall_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )
        metrics['f1'] = f1_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        metrics['confusion_matrix'] = cm
        
        # Binary classification specific metrics
        if self.is_binary:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Additional binary metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = metrics['recall']  # Same as recall
            metrics['balanced_accuracy'] = (metrics['specificity'] + metrics['sensitivity']) / 2
            
            # Matthews Correlation Coefficient
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if mcc_denominator > 0:
                metrics['mcc'] = (tp * tn - fp * fn) / mcc_denominator
            else:
                metrics['mcc'] = 0
        
        # Probability-based metrics
        if self.y_proba is not None:
            if self.is_binary:
                # Use probabilities for positive class
                y_proba_pos = self.y_proba[:, self.pos_label] if self.y_proba.ndim > 1 else self.y_proba
                
                metrics['roc_auc'] = roc_auc_score(self.y_true, y_proba_pos)
                metrics['average_precision'] = average_precision_score(self.y_true, y_proba_pos)
                metrics['log_loss'] = log_loss(self.y_true, self.y_proba)
                
                # Brier score
                metrics['brier_score'] = np.mean((y_proba_pos - self.y_true) ** 2)
                
                # Calibration metrics
                metrics.update(self._calculate_calibration_metrics(y_proba_pos))
                
            else:  # Multiclass
                # One-vs-rest ROC AUC
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        self.y_true, self.y_proba, multi_class='ovr'
                    )
                    metrics['roc_auc_ovo'] = roc_auc_score(
                        self.y_true, self.y_proba, multi_class='ovo'
                    )
                except:
                    logger.warning("Could not calculate multiclass ROC AUC")
                
                metrics['log_loss'] = log_loss(self.y_true, self.y_proba)
        
        # Per-class metrics
        if not self.is_binary:
            metrics['per_class_metrics'] = self._calculate_per_class_metrics()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            self.y_true, self.y_pred, output_dict=True
        )
        
        return metrics
    
    def _calculate_calibration_metrics(self, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate probability calibration metrics"""
        from sklearn.calibration import calibration_curve
        
        calibration_metrics = {}
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        ece = 0
        for i in range(n_bins):
            mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = self.y_true[mask].mean()
                bin_conf = y_proba[mask].mean()
                bin_weight = mask.sum() / len(y_proba)
                ece += bin_weight * abs(bin_acc - bin_conf)
        
        calibration_metrics['expected_calibration_error'] = ece
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for i in range(n_bins):
            mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = self.y_true[mask].mean()
                bin_conf = y_proba[mask].mean()
                mce = max(mce, abs(bin_acc - bin_conf))
        
        calibration_metrics['max_calibration_error'] = mce
        
        return calibration_metrics
    
    def _calculate_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each class"""
        per_class = {}
        
        for class_label in np.unique(self.y_true):
            # Create binary problem for this class
            y_true_binary = (self.y_true == class_label).astype(int)
            y_pred_binary = (self.y_pred == class_label).astype(int)
            
            per_class[f'class_{class_label}'] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': int((self.y_true == class_label).sum())
            }
        
        return per_class
    
    def get_metrics(self, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get specific metrics or all metrics"""
        if subset is None:
            return self.metrics
        
        return {k: v for k, v in self.metrics.items() if k in subset}
    
    def get_curve_data(self) -> Dict[str, Any]:
        """Get data for ROC and PR curves"""
        if self.y_proba is None or not self.is_binary:
            return {}
        
        y_proba_pos = self.y_proba[:, self.pos_label] if self.y_proba.ndim > 1 else self.y_proba
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, y_proba_pos)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(self.y_true, y_proba_pos)
        
        return {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': self.metrics.get('roc_auc', 0)
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist(),
                'auc': self.metrics.get('average_precision', 0)
            }
        }


class RegressionMetrics:
    """Calculate and store regression metrics"""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 sample_weight: Optional[np.ndarray] = None):
        """
        Initialize regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weight: Optional sample weights
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weight = sample_weight
        
        # Calculate residuals
        self.residuals = y_true - y_pred
        
        # Calculate all metrics
        self.metrics = self._calculate_all_metrics()
        
    def _calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all regression metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )
        metrics['r2'] = r2_score(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )
        
        # Additional metrics
        if not np.any(self.y_true == 0):  # Avoid division by zero
            metrics['mape'] = mean_absolute_percentage_error(self.y_true, self.y_pred)
        else:
            # Calculate MAPE only for non-zero true values
            mask = self.y_true != 0
            if mask.sum() > 0:
                metrics['mape'] = mean_absolute_percentage_error(
                    self.y_true[mask], self.y_pred[mask]
                )
            else:
                metrics['mape'] = np.nan
        
        metrics['median_ae'] = median_absolute_error(self.y_true, self.y_pred)
        metrics['explained_variance'] = explained_variance_score(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )
        
        # Residual statistics
        metrics['mean_residual'] = np.mean(self.residuals)
        metrics['std_residual'] = np.std(self.residuals)
        metrics['min_residual'] = np.min(self.residuals)
        metrics['max_residual'] = np.max(self.residuals)
        metrics['median_residual'] = np.median(self.residuals)
        
        # Percentile errors
        abs_errors = np.abs(self.residuals)
        metrics['p50_error'] = np.percentile(abs_errors, 50)
        metrics['p90_error'] = np.percentile(abs_errors, 90)
        metrics['p95_error'] = np.percentile(abs_errors, 95)
        metrics['p99_error'] = np.percentile(abs_errors, 99)
        
        # Directional metrics
        metrics['overestimation_rate'] = np.mean(self.y_pred > self.y_true)
        metrics['underestimation_rate'] = np.mean(self.y_pred < self.y_true)
        
        # Within threshold metrics (for wins prediction)
        metrics['within_5'] = np.mean(abs_errors <= 5)
        metrics['within_10'] = np.mean(abs_errors <= 10)
        
        # Adjusted R² (assuming we know the number of features)
        n = len(self.y_true)
        # Note: p (number of features) should be passed in for accurate calculation
        # Using placeholder value
        p = 10  
        if n > p + 1:
            metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        else:
            metrics['adjusted_r2'] = metrics['r2']
        
        return metrics
    
    def get_residual_analysis(self) -> Dict[str, Any]:
        """Perform residual analysis"""
        from scipy import stats
        
        analysis = {}
        
        # Normality test
        if len(self.residuals) >= 8:
            statistic, p_value = stats.normaltest(self.residuals)
            analysis['normality_test'] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        
        # Autocorrelation (Durbin-Watson)
        if len(self.residuals) > 1:
            diff = np.diff(self.residuals)
            dw = np.sum(diff ** 2) / np.sum(self.residuals ** 2)
            analysis['durbin_watson'] = float(dw)
        
        # Heteroscedasticity test (simplified Breusch-Pagan)
        if len(self.residuals) >= 10:
            # Regress squared residuals on predictions
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(self.y_pred.reshape(-1, 1), self.residuals ** 2)
            r2_resid = lr.score(self.y_pred.reshape(-1, 1), self.residuals ** 2)
            
            # Simplified test statistic
            lm = len(self.residuals) * r2_resid
            analysis['heteroscedasticity'] = {
                'lm_statistic': float(lm),
                'r2_residuals': float(r2_resid),
                'likely_heteroscedastic': r2_resid > 0.1
            }
        
        # Outliers (using IQR method)
        q1 = np.percentile(self.residuals, 25)
        q3 = np.percentile(self.residuals, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (self.residuals < lower_bound) | (self.residuals > upper_bound)
        analysis['outliers'] = {
            'n_outliers': int(outliers.sum()),
            'outlier_percentage': float(outliers.mean() * 100),
            'outlier_indices': np.where(outliers)[0].tolist()
        }
        
        return analysis


# Convenience functions
def calculate_classification_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_proba: Optional[np.ndarray] = None,
                                   metrics_list: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Quick function to calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        metrics_list: List of metrics to calculate
        
    Returns:
        Dictionary of metrics
    """
    cm = ClassificationMetrics(y_true, y_pred, y_proba)
    
    if metrics_list is None:
        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    return cm.get_metrics(metrics_list)


def calculate_regression_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               metrics_list: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Quick function to calculate regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metrics_list: List of metrics to calculate
        
    Returns:
        Dictionary of metrics
    """
    rm = RegressionMetrics(y_true, y_pred)
    
    if metrics_list is None:
        metrics_list = ['rmse', 'mae', 'r2', 'mape']
    
    return {k: v for k, v in rm.metrics.items() if k in metrics_list}


def calculate_milestone_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              milestone_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for multiple milestone predictions
    
    Args:
        y_true: True milestone achievements (n_samples, n_milestones)
        y_pred: Predicted milestone achievements
        milestone_names: Names of milestones
        
    Returns:
        Dictionary of metrics per milestone
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    results = {}
    
    for i, milestone in enumerate(milestone_names):
        cm = ClassificationMetrics(y_true[:, i], y_pred[:, i])
        results[milestone] = {
            'accuracy': cm.metrics['accuracy'],
            'precision': cm.metrics['precision'],
            'recall': cm.metrics['recall'],
            'f1': cm.metrics['f1']
        }
    
    # Overall metrics
    overall_accuracy = np.mean([results[m]['accuracy'] for m in milestone_names])
    results['overall'] = {'accuracy': overall_accuracy}
    
    return results


# Custom scorers for sklearn
def make_custom_scorer(metric_name: str, greater_is_better: bool = True) -> Any:
    """
    Create custom scorer for cross-validation
    
    Args:
        metric_name: Name of the metric
        greater_is_better: Whether higher values are better
        
    Returns:
        Scorer function
    """
    def scorer(y_true, y_pred):
        if metric_name == 'within_5_wins':
            return np.mean(np.abs(y_true - y_pred) <= 5)
        elif metric_name == 'within_10_wins':
            return np.mean(np.abs(y_true - y_pred) <= 10)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    return make_scorer(scorer, greater_is_better=greater_is_better)


if __name__ == "__main__":
    # Test classification metrics
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    y_proba = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9],
                        [0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
    
    cm = ClassificationMetrics(y_true, y_pred, y_proba)
    print("Classification Metrics:")
    for key, value in cm.metrics.items():
        if not isinstance(value, (np.ndarray, dict)):
            print(f"  {key}: {value:.4f}")
    
    # Test regression metrics
    y_true_reg = np.array([85, 92, 78, 95, 88, 82, 90, 87])
    y_pred_reg = np.array([82, 90, 80, 92, 85, 85, 88, 89])
    
    rm = RegressionMetrics(y_true_reg, y_pred_reg)
    print("\nRegression Metrics:")
    for key, value in rm.metrics.items():
        print(f"  {key}: {value:.4f}")