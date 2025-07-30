"""
Classification models for predicting division winners and playoff teams
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Optional, Union, List
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class DivisionWinnerClassifier(BaseModel):
    """Classification model for predicting division winners"""
    
    def __init__(self, model_type: str = 'random_forest', 
                 calibrate_probabilities: bool = False,
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize classifier
        
        Args:
            model_type: Type of classification model
            calibrate_probabilities: Whether to calibrate predicted probabilities
            model_params: Optional parameters for the model
        """
        super().__init__(model_name=f"DivisionWinner_{model_type}")
        
        self.model_type = model_type
        self.calibrate_probabilities = calibrate_probabilities
        self.model_params = model_params or {}
        
        # Initialize model
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the specified model type"""
        # Default parameters for each model type
        default_params = {
            'logistic': {
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            },
            'gradient_boost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'subsample': 0.8
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'objective': 'binary:logistic',
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': 42,
                'objective': 'binary',
                'metric': 'binary_logloss',
                'force_col_wise': True
            },
            'svm': {
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'neural_network': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        }
        
        # Get model class and parameters
        model_classes = {
            'logistic': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boost': GradientBoostingClassifier,
            'xgboost': xgb.XGBClassifier,
            'lightgbm': lgb.LGBMClassifier,
            'svm': SVC,
            'neural_network': MLPClassifier
        }
        
        if self.model_type not in model_classes:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Choose from: {list(model_classes.keys())}")
        
        # Merge default and custom parameters
        params = default_params.get(self.model_type, {})
        params.update(self.model_params)
        
        # Create model instance
        model = model_classes[self.model_type](**params)
        
        # Apply probability calibration if requested
        if self.calibrate_probabilities:
            model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None):
        """
        Train the classification model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        logger.info(f"Training {self.model_type} classifier...")
        
        # Special handling for models that support validation sets
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            if self.model_type == 'xgboost':
                eval_set = [(X_val, y_val)]
                self.model.fit(X_train, y_train, eval_set=eval_set, 
                              early_stopping_rounds=20, verbose=False)
            else:  # lightgbm
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              early_stopping_rounds=20, verbose=False)
        else:
            # Standard fit
            self.model.fit(X_train, y_train)
        
        # Log training results
        train_score = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        if X_val is not None:
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions"""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for each class"""
        X = self.validate_input(X)
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: np.ndarray, 
                               confidence_threshold: float = 0.7) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence levels
        
        Args:
            X: Features for prediction
            confidence_threshold: Threshold for high confidence predictions
            
        Returns:
            Dictionary with predictions, probabilities, and confidence levels
        """
        X = self.validate_input(X)
        
        # Get probabilities
        probas = self.predict_proba(X)
        
        # Get predictions
        predictions = (probas[:, 1] >= 0.5).astype(int)
        
        # Calculate confidence (distance from 0.5)
        confidence = np.abs(probas[:, 1] - 0.5) * 2
        
        # Categorize confidence levels
        confidence_levels = np.where(
            confidence >= confidence_threshold,
            'high',
            np.where(confidence >= 0.4, 'medium', 'low')
        )
        
        return {
            'predictions': predictions,
            'probabilities': probas[:, 1],
            'confidence': confidence,
            'confidence_levels': confidence_levels
        }
    
    def get_model_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, log_loss
        )
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'average_precision': average_precision_score(y_test, y_proba),
            'log_loss': log_loss(y_test, y_proba)
        }
        
        # Add model-specific metrics
        if self.model_type in ['random_forest', 'gradient_boost', 'xgboost', 'lightgbm']:
            # Feature importance available
            importance_df = self.get_feature_importance()
            if importance_df is not None:
                metrics['top_feature'] = importance_df.iloc[0]['feature']
                metrics['top_feature_importance'] = importance_df.iloc[0]['importance']
        
        return metrics


class PlayoffClassifier(DivisionWinnerClassifier):
    """Specialized classifier for playoff prediction"""
    
    def __init__(self, model_type: str = 'xgboost', 
                 playoff_spots: int = 10,
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize playoff classifier
        
        Args:
            model_type: Type of classification model
            playoff_spots: Number of playoff spots available
            model_params: Optional model parameters
        """
        super().__init__(model_type=model_type, 
                        calibrate_probabilities=True,
                        model_params=model_params)
        
        self.model_name = f"Playoff_{model_type}"
        self.playoff_spots = playoff_spots
    
    def predict_playoff_probability(self, X: np.ndarray, 
                                   current_standings: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Predict playoff probability with optional standings adjustment
        
        Args:
            X: Features for prediction
            current_standings: Current division standings for context
            
        Returns:
            Playoff probabilities
        """
        base_probas = self.predict_proba(X)[:, 1]
        
        if current_standings is not None:
            # Adjust probabilities based on current standings
            # This is simplified - real implementation would be more complex
            games_behind = current_standings.get('games_behind', 0)
            
            # Reduce probability for teams far behind
            adjustment = 1 / (1 + games_behind / 10)
            adjusted_probas = base_probas * adjustment
            
            # Normalize to maintain probability properties
            adjusted_probas = adjusted_probas / adjusted_probas.sum() * base_probas.sum()
            
            return adjusted_probas
        
        return base_probas


class MultiClassTeamClassifier(BaseModel):
    """Multi-class classifier for team performance tiers"""
    
    def __init__(self, model_type: str = 'random_forest',
                 n_classes: int = 5,
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-class classifier
        
        Args:
            model_type: Type of classification model
            n_classes: Number of performance tiers
            model_params: Optional model parameters
        """
        super().__init__(model_name=f"TeamTier_{model_type}")
        
        self.model_type = model_type
        self.n_classes = n_classes
        self.model_params = model_params or {}
        
        # Initialize model
        self.model = self._initialize_multiclass_model()
        
        # Define tier labels
        self.tier_labels = {
            0: 'Rebuilding',
            1: 'Below Average',
            2: 'Average',
            3: 'Contender',
            4: 'Elite'
        }
    
    def _initialize_multiclass_model(self):
        """Initialize multi-class model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                **self.model_params
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                objective='multi:softprob',
                num_class=self.n_classes,
                random_state=42,
                **self.model_params
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train multi-class classifier"""
        # Convert continuous targets to classes if needed
        if len(np.unique(y_train)) > self.n_classes:
            y_train = self._create_performance_tiers(y_train)
        
        self.model.fit(X_train, y_train)
    
    def _create_performance_tiers(self, wins: np.ndarray) -> np.ndarray:
        """Convert win totals to performance tiers"""
        # Define tier boundaries (can be adjusted based on era)
        boundaries = [0, 70, 81, 90, 95, 999]
        
        tiers = np.digitize(wins, boundaries) - 1
        tiers = np.clip(tiers, 0, self.n_classes - 1)
        
        return tiers
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict team performance tier"""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def predict_tier_names(self, X: np.ndarray) -> List[str]:
        """Predict tier names instead of numeric classes"""
        predictions = self.predict(X)
        return [self.tier_labels[pred] for pred in predictions]