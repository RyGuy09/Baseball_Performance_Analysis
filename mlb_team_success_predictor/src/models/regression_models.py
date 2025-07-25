"""
Regression models for predicting win totals and run production
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Optional, Tuple
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class WinsRegressor(BaseModel):
    """Regression model for predicting season win totals"""
    
    def __init__(self, model_type: str = 'random_forest',
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize regression model
        
        Args:
            model_type: Type of regression model
            model_params: Optional model parameters
        """
        super().__init__(model_name=f"WinsRegressor_{model_type}")
        
        self.model_type = model_type
        self.model_params = model_params or {}
        
        # Initialize model
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the specified regression model"""
        # Default parameters for each model type
        default_params = {
            'linear': {
                'fit_intercept': True
            },
            'ridge': {
                'alpha': 1.0,
                'random_state': 42
            },
            'lasso': {
                'alpha': 1.0,
                'random_state': 42,
                'max_iter': 2000
            },
            'elastic_net': {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'random_state': 42,
                'max_iter': 2000
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
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
                'objective': 'reg:squarederror'
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse',
                'force_col_wise': True
            },
            'svr': {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1
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
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elastic_net': ElasticNet,
            'random_forest': RandomForestRegressor,
            'gradient_boost': GradientBoostingRegressor,
            'xgboost': xgb.XGBRegressor,
            'lightgbm': lgb.LGBMRegressor,
            'svr': SVR,
            'neural_network': MLPRegressor
        }
        
        if self.model_type not in model_classes:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Choose from: {list(model_classes.keys())}")
        
        # Merge default and custom parameters
        params = default_params.get(self.model_type, {})
        params.update(self.model_params)
        
        # Create model instance
        return model_classes[self.model_type](**params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None):
        """
        Train the regression model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        logger.info(f"Training {self.model_type} regressor...")
        
        # Special handling for models that support validation sets
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            if self.model_type == 'xgboost':
                eval_set = [(X_val, y_val)]
                self.model.fit(X_train, y_train, eval_set=eval_set,
                              early_stopping_rounds=20, verbose=False)
            else:  # lightgbm
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        else:
            # Standard fit
            self.model.fit(X_train, y_train)
        
        # Log training results
        train_pred = self.predict(X_train)
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        logger.info(f"Training RMSE: {train_rmse:.4f}")
        
        if X_val is not None:
            val_pred = self.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            logger.info(f"Validation RMSE: {val_rmse:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict win totals"""
        X = self.validate_input(X)
        predictions = self.model.predict(X)
        
        # Clip predictions to valid range (0-162 games)
        predictions = np.clip(predictions, 0, 162)
        
        return predictions
    
    def predict_with_bounds(self, X: np.ndarray, 
                           confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Predict with confidence bounds
        
        Args:
            X: Features for prediction
            confidence_level: Confidence level for bounds
            
        Returns:
            Dictionary with predictions and bounds
        """
        X = self.validate_input(X)
        
        # Get point predictions
        predictions = self.predict(X)
        
        # Calculate prediction intervals
        if self.model_type in ['random_forest', 'gradient_boost']:
            # Use tree-based prediction intervals
            bounds = self._tree_prediction_intervals(X, confidence_level)
        else:
            # Use residual-based intervals (simplified)
            # In practice, you'd calculate this from validation residuals
            std_error = 10.0  # Approximate standard error
            z_score = 1.96 if confidence_level == 0.95 else 2.58
            
            bounds = {
                'lower': predictions - z_score * std_error,
                'upper': predictions + z_score * std_error
            }
        
        # Clip bounds to valid range
        bounds['lower'] = np.clip(bounds['lower'], 0, 162)
        bounds['upper'] = np.clip(bounds['upper'], 0, 162)
        
        return {
            'predictions': predictions,
            'lower_bound': bounds['lower'],
            'upper_bound': bounds['upper'],
            'confidence_level': confidence_level
        }
    
    def _tree_prediction_intervals(self, X: np.ndarray, 
                                  confidence_level: float) -> Dict[str, np.ndarray]:
        """Calculate prediction intervals for tree-based models"""
        if not hasattr(self.model, 'estimators_'):
            # Fallback for non-ensemble models
            return {'lower': np.zeros(len(X)), 'upper': np.ones(len(X)) * 162}
        
        # Get predictions from all trees
        all_predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate percentiles
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(all_predictions, alpha * 100, axis=0)
        upper = np.percentile(all_predictions, (1 - alpha) * 100, axis=0)
        
        return {'lower': lower, 'upper': upper}
    
    def get_model_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mean_error': np.mean(y_pred - y_test),
            'std_error': np.std(y_pred - y_test)
        }
        
        # Add percentile metrics
        errors = np.abs(y_pred - y_test)
        metrics['median_absolute_error'] = np.median(errors)
        metrics['p90_error'] = np.percentile(errors, 90)
        metrics['within_5_wins'] = np.mean(errors <= 5)
        metrics['within_10_wins'] = np.mean(errors <= 10)
        
        return metrics


class RunProductionRegressor(WinsRegressor):
    """Specialized regressor for run production prediction"""
    
    def __init__(self, target_type: str = 'runs_scored',
                 model_type: str = 'xgboost',
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize run production regressor
        
        Args:
            target_type: 'runs_scored' or 'runs_allowed'
            model_type: Type of regression model
            model_params: Optional model parameters
        """
        super().__init__(model_type=model_type, model_params=model_params)
        
        self.model_name = f"RunProduction_{target_type}_{model_type}"
        self.target_type = target_type
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict run production"""
        X = self.validate_input(X)
        predictions = self.model.predict(X)
        
        # Clip predictions to reasonable range
        # Modern teams typically score between 500-1000 runs
        predictions = np.clip(predictions, 400, 1100)
        
        return predictions
    
    def predict_run_differential(self, X: np.ndarray,
                                runs_scored_model: Optional['RunProductionRegressor'] = None,
                                runs_allowed_model: Optional['RunProductionRegressor'] = None) -> np.ndarray:
        """
        Predict run differential using separate models
        
        Args:
            X: Features
            runs_scored_model: Model for runs scored (if different from self)
            runs_allowed_model: Model for runs allowed (if different from self)
            
        Returns:
            Predicted run differential
        """
        if self.target_type == 'runs_scored':
            runs_scored = self.predict(X)
            if runs_allowed_model is not None:
                runs_allowed = runs_allowed_model.predict(X)
            else:
                # Estimate based on typical relationship
                runs_allowed = 750 + (750 - runs_scored) * 0.5
        else:  # runs_allowed
            runs_allowed = self.predict(X)
            if runs_scored_model is not None:
                runs_scored = runs_scored_model.predict(X)
            else:
                # Estimate based on typical relationship
                runs_scored = 750 + (750 - runs_allowed) * 0.5
        
        return runs_scored - runs_allowed


class PythagoreanRegressor(BaseModel):
    """Specialized model using Pythagorean expectation"""
    
    def __init__(self, exponent: float = 2.0, use_dynamic_exponent: bool = True):
        """
        Initialize Pythagorean model
        
        Args:
            exponent: Pythagorean exponent (typically around 2)
            use_dynamic_exponent: Whether to fit exponent to data
        """
        super().__init__(model_name="PythagoreanRegressor")
        
        self.exponent = exponent
        self.use_dynamic_exponent = use_dynamic_exponent
        self.fitted_exponent = exponent
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit Pythagorean model
        
        Expects X_train to have runs_scored and runs_allowed as features
        """
        # Extract runs scored and allowed
        # Assumes these are the first two features
        runs_scored = X_train[:, 0]
        runs_allowed = X_train[:, 1]
        
        if self.use_dynamic_exponent:
            # Optimize exponent
            from scipy.optimize import minimize_scalar
            
            def loss(exp):
                pred_wins = self._pythagorean_wins(runs_scored, runs_allowed, exp)
                return np.mean((y_train - pred_wins) ** 2)
            
            result = minimize_scalar(loss, bounds=(1.5, 3.0), method='bounded')
            self.fitted_exponent = result.x
            
            logger.info(f"Fitted Pythagorean exponent: {self.fitted_exponent:.3f}")
        
        self.model = lambda X: self._pythagorean_wins(X[:, 0], X[:, 1], self.fitted_exponent)
    
    def _pythagorean_wins(self, runs_scored: np.ndarray, 
                         runs_allowed: np.ndarray, 
                         exponent: float) -> np.ndarray:
        """Calculate Pythagorean expected wins"""
        rs_exp = runs_scored ** exponent
        ra_exp = runs_allowed ** exponent
        
        # Assume 162 game season
        pythagorean_pct = rs_exp / (rs_exp + ra_exp)
        return 162 * pythagorean_pct
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Pythagorean formula"""
        X = self.validate_input(X)
        return self.model(X)