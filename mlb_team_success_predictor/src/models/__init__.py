"""
Machine Learning models for MLB Team Success Predictor

This module contains:
- Base model class with common functionality
- Classification models for division winner prediction
- Regression models for win total prediction
- Milestone prediction models
- Ensemble methods for improved accuracy
- Time series models for historical analysis
"""

from .base_model import BaseModel
from .classification_models import DivisionWinnerClassifier
from .regression_models import WinsRegressor
from .milestone_predictor import MilestonePredictor
from .ensemble_models import MLBEnsembleModel
from .time_series_models import TeamPerformanceForecaster

__all__ = [
    'BaseModel',
    'DivisionWinnerClassifier',
    'WinsRegressor',
    'MilestonePredictor',
    'MLBEnsembleModel',
    'TeamPerformanceForecaster'
]

# Model version for tracking
MODEL_VERSION = '1.0.0'