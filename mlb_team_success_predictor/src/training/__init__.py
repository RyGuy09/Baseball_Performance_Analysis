"""
Training module for MLB Team Success Predictor

This module handles:
- Model training pipelines
- Hyperparameter optimization
- Cross-validation strategies
- Training orchestration
"""

from .train_classifier import ClassifierTrainer
from .train_regressor import RegressorTrainer
from .hyperparameter_tuning import HyperparameterTuner
from .cross_validation import CrossValidator

__all__ = [
    'ClassifierTrainer',
    'RegressorTrainer',
    'HyperparameterTuner',
    'CrossValidator'
]

# Training configuration defaults
DEFAULT_TRAIN_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': True
}