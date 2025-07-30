"""
Evaluation module for MLB Team Success Predictor

This module handles:
- Model performance metrics
- Comprehensive model evaluation
- Performance visualization
- Model comparison
"""

from .metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_milestone_metrics
)

from .model_evaluation import (
    ModelEvaluator,
    ClassificationEvaluator,
    RegressionEvaluator,
    EnsembleEvaluator,
    create_evaluation_report
)

__all__ = [
    # Metrics
    'ClassificationMetrics',
    'RegressionMetrics',
    'calculate_classification_metrics',
    'calculate_regression_metrics',
    'calculate_milestone_metrics',
    
    # Evaluators
    'ModelEvaluator',
    'ClassificationEvaluator',
    'RegressionEvaluator',
    'EnsembleEvaluator',
    'create_evaluation_report'
]

# Default evaluation settings
DEFAULT_EVAL_CONFIG = {
    'classification_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'regression_metrics': ['rmse', 'mae', 'r2', 'mape'],
    'confidence_level': 0.95,
    'bootstrap_iterations': 1000,
    'save_plots': True
}