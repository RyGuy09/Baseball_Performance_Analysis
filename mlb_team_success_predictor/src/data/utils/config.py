"""
Configuration management for MLB Team Success Predictor
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
LOG_DIR = PROJECT_ROOT / 'logs'

# Data paths
RAW_DATA_PATH = DATA_DIR / 'raw' / 'mlb_stats_1901_to_2025.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'processed'
EXTERNAL_DATA_PATH = DATA_DIR / 'external'

# Model paths
SAVED_MODELS_PATH = MODEL_DIR / 'saved_models'
SCALERS_PATH = MODEL_DIR / 'scalers'
ARTIFACTS_PATH = MODEL_DIR / 'artifacts'

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, PROCESSED_DATA_PATH, 
                  SAVED_MODELS_PATH, SCALERS_PATH, ARTIFACTS_PATH]:
    directory.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_MODELS = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
REGRESSION_MODELS = ['linear_regression', 'random_forest', 'xgboost', 'lightgbm']


class Config:
    """Configuration class for managing settings"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or CONFIG_DIR / 'config.yaml'
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix == '.yaml':
                    return yaml.safe_load(f)
                elif self.config_path.suffix == '.json':
                    return json.load(f)
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'data': {
                'era_strategy': 'modern',
                'min_year': 2006,
                'max_year': 2024,
                'test_size': 0.2,
                'validation_size': 0.1,
                'random_state': 42
            },
            'features': {
                'classification': CLASSIFICATION_FEATURES,
                'regression': REGRESSION_FEATURES,
                'milestone': MILESTONE_FEATURES
            },
            'models': {
                'classification_types': ['random_forest', 'xgboost', 'lightgbm'],
                'regression_types': ['random_forest', 'xgboost', 'elastic_net'],
                'use_calibration': True,
                'use_ensemble': True
            },
            'training': {
                'cv_folds': 5,
                'scoring_metric_classification': 'roc_auc',
                'scoring_metric_regression': 'neg_mean_squared_error',
                'n_jobs': -1,
                'verbose': 1
            },
            'hyperparameters': {
                'search_method': 'optuna',
                'n_trials': 50,
                'timeout': 3600  # 1 hour
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[Path] = None):
        """Save configuration to file"""
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            if save_path.suffix == '.yaml':
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")


# Feature lists
CLASSIFICATION_FEATURES = [
    # Current performance
    'wins', 'losses', 'winning_percentage', 'run_differential',
    
    # Home/Away splits
    'home_win_pct', 'away_win_pct', 'home_away_diff',
    
    # Run metrics
    'runs_per_game', 'runs_allowed_per_game', 'run_efficiency',
    
    # Expected performance
    'pythagorean_wins', 'pythagorean_win_pct', 'luck_factor',
    
    # Historical features
    'prev_wins', 'prev_run_differential', 'prev_winning_percentage',
    'wins_3yr_avg', 'run_differential_3yr_avg',
    
    # Trends
    'wins_trend', 'momentum',
    
    # Strength indicators
    'relative_strength', 'wins_percentile'
]

REGRESSION_FEATURES = [
    # Core predictors
    'home_win_pct', 'away_win_pct', 'run_differential',
    'runs_per_game', 'runs_allowed_per_game',
    
    # Expected metrics
    'pythagorean_wins', 'pythagorean_win_pct',
    
    # Historical performance
    'prev_wins', 'prev_run_differential',
    'wins_3yr_avg', 'wins_5yr_avg',
    'run_differential_3yr_avg',
    
    # Trends and consistency
    'wins_trend', 'run_differential_trend',
    'performance_volatility', 'consistency_score',
    
    # Context
    'year', 'competitive_balance'
]

MILESTONE_FEATURES = [
    'winning_percentage', 'run_differential',
    'home_win_pct', 'away_win_pct',
    'runs_per_game', 'runs_allowed_per_game',
    'pythagorean_wins', 'pythagorean_win_pct',
    'relative_strength', 'wins_percentile',
    'momentum', 'consistency_score'
]

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    },
    'lightgbm': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'random_state': 42,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'force_col_wise': True,
        'verbose': -1
    },
    'logistic': {
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'liblinear'
    },
    'elastic_net': {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'max_iter': 2000,
        'random_state': 42
    }
}

# Training parameters
TRAINING_PARAMS = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'cv_folds': 5,
    'random_state': 42,
    'stratify': True,
    'shuffle': True
}

# Evaluation metrics
CLASSIFICATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'
]

REGRESSION_METRICS = [
    'rmse', 'mae', 'r2', 'mape', 'median_absolute_error'
]


# Singleton config instance
_config_instance = None

def get_config(config_path: Optional[Path] = None) -> Config:
    """Get or create config instance"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance

def load_config(config_path: Path) -> Config:
    """Load a new config instance"""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance


# Convenience functions
def get_feature_config(model_type: str) -> List[str]:
    """Get features for a specific model type"""
    if 'classification' in model_type.lower():
        return CLASSIFICATION_FEATURES
    elif 'regression' in model_type.lower():
        return REGRESSION_FEATURES
    elif 'milestone' in model_type.lower():
        return MILESTONE_FEATURES
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_params(model_name: str) -> Dict[str, Any]:
    """Get default parameters for a model"""
    return MODEL_PARAMS.get(model_name, {})


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    
    print("Project Structure:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Model Dir: {MODEL_DIR}")
    
    print("\nConfiguration:")
    print(f"  Era Strategy: {config.get('data.era_strategy')}")
    print(f"  CV Folds: {config.get('training.cv_folds')}")
    
    print("\nFeature Counts:")
    print(f"  Classification: {len(CLASSIFICATION_FEATURES)}")
    print(f"  Regression: {len(REGRESSION_FEATURES)}")
    print(f"  Milestone: {len(MILESTONE_FEATURES)}")