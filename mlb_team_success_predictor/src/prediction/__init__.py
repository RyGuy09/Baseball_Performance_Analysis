"""
Prediction module for MLB Team Success Predictor

This module handles:
- Model loading and prediction
- Prediction pipelines
- Batch predictions
- Real-time predictions
"""

from .predictor import (
    MLBPredictor,
    DivisionWinnerPredictor,
    WinsPredictor,
    MilestonePredictor,
    EnsemblePredictor
)

from .prediction_pipeline import (
    PredictionPipeline,
    BatchPredictionPipeline,
    RealTimePredictor,
    create_prediction_pipeline
)

__all__ = [
    # Predictors
    'MLBPredictor',
    'DivisionWinnerPredictor',
    'WinsPredictor',
    'MilestonePredictor',
    'EnsemblePredictor',
    
    # Pipelines
    'PredictionPipeline',
    'BatchPredictionPipeline',
    'RealTimePredictor',
    'create_prediction_pipeline'
]

# Default prediction settings
DEFAULT_PREDICTION_CONFIG = {
    'confidence_level': 0.95,
    'include_probabilities': True,
    'include_explanations': False,
    'batch_size': 1000
}