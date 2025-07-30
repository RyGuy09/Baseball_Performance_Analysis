"""
MLB Team Success Predictor

A machine learning framework for predicting MLB team performance including:
- Division winner classification
- Season win totals regression
- Milestone achievement prediction
- Era-aware historical analysis

Author: [Your Name]
Date: 2024
License: MIT
"""

# Version info
__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

# Import main components for easier access
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models.classification_models import DivisionWinnerClassifier
from .models.regression_models import WinsRegressor
from .models.milestone_predictor import MilestonePredictor
from .prediction.predictor import MLBPredictor
from .utils.config import (
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
    MILESTONE_FEATURES,
    MODEL_PARAMS
)

# Define what should be imported with "from src import *"
__all__ = [
    'DataLoader',
    'FeatureEngineer',
    'DivisionWinnerClassifier',
    'WinsRegressor',
    'MilestonePredictor',
    'MLBPredictor',
    'CLASSIFICATION_FEATURES',
    'REGRESSION_FEATURES',
    'MILESTONE_FEATURES',
    'MODEL_PARAMS'
]

# Package metadata
PACKAGE_NAME = 'mlb_team_success_predictor'
DESCRIPTION = 'Machine learning models for predicting MLB team success metrics'

# Logging configuration
import logging
import sys

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only add handler if no handlers exist (prevents duplicate logs)
if not logger.handlers:
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)

# Welcome message when package is imported
logger.debug(f"Initializing {PACKAGE_NAME} v{__version__}")