"""
Ensemble models combining multiple predictors for improved accuracy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

from .base_model import BaseModel
from .classification_models import DivisionWinnerClassifier
from .regression_models import WinsRegressor

logger = logging.getLogger(__name__)


class MLBEnsembleModel(BaseModel):
    """Ensemble model combining multiple MLB predictors"""
    
    def __init__(self, ensemble_type: str = 'voting',
                 task_type: str = 'classification',
                 models: Optional[List[Tuple[str, Any]]] = None,
                 meta_learner: Optional[Any] = None,
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble model
        
        Args:
            ensemble_type: 'voting', 'stacking', or 'blending'
            task_type: 'classification' or 'regression'
            models: List of (name, model) tuples
            meta_learner: Meta-learner for stacking
            weights: Weights for voting ensemble
        """
        super().__init__(model_name=f"Ensemble_{ensemble_type}_{task_type}")
        
        self.ensemble_type = ensemble_type
        self.task_type = task_type
        self.weights = weights
        
        # Initialize models if not provided
        if models is None:
            self.base_models = self._create_default_models()
        else:
            self.base_models = models
        
        # Initialize meta-learner
        if meta_learner is None and ensemble_type == 'stacking':
            self.meta_learner = self._create_default_meta_learner()
        else:
            self.meta_learner = meta_learner
        
        # Create ensemble
        self.model = self._create_ensemble()
        
    def _create_default_models(self) -> List[Tuple[str, Any]]:
        """Create default base models"""
        if self.task_type == 'classification':
            return [
                ('rf', DivisionWinnerClassifier('random_forest').model),
                ('xgb', DivisionWinnerClassifier('xgboost').model),
                ('lgb', DivisionWinnerClassifier('lightgbm').model),
            ]
        else:  # regression
            return [
                ('rf', WinsRegressor('random_forest').model),
                ('xgb', WinsRegressor('xgboost').model),
                ('lgb', WinsRegressor('lightgbm').model),
            ]
    
    def _create_default_meta_learner(self):
        """Create default meta-learner for stacking"""
        if self.task_type == 'classification':
            return LogisticRegression(random_state=42)
        else:
            return Ridge(alpha=1.0, random_state=42)
    
    def _create_ensemble(self):
        """Create the ensemble model"""
        if self.ensemble_type == 'voting':
            if self.task_type == 'classification':
                return VotingClassifier(
                    estimators=self.base_models,
                    voting='soft',
                    weights=self.weights
                )
            else:
                return VotingRegressor(
                    estimators=self.base_models,
                    weights=self.weights
                )
        
        elif self.ensemble_type == 'stacking':
            if self.task_type == 'classification':
                return StackingClassifier(
                    estimators=self.base_models,
                    final_estimator=self.meta_learner,
                    cv=5,
                    stack_method='predict_proba',
                    n_jobs=-1
                )
            else:
                return StackingRegressor(
                    estimators=self.base_models,
                    final_estimator=self.meta_learner,
                    cv=5,
                    n_jobs=-1
                )
        
        elif self.ensemble_type == 'blending':
            # Custom blending implementation
            return BlendingEnsemble(
                base_models=self.base_models,
                meta_learner=self.meta_learner,
                task_type=self.task_type
            )
        
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None):
        """Train ensemble model"""
        logger.info(f"Training {self.ensemble_type} ensemble...")
        
        if self.ensemble_type == 'blending' and X_val is not None:
            # Blending requires validation set
            self.model.fit(X_train, y_train, X_val, y_val)
        else:
            self.model.fit(X_train, y_train)
        
        # Log ensemble performance
        train_score = self.model.score(X_train, y_train)
        metric_name = "accuracy" if self.task_type == "classification" else "R²"
        logger.info(f"Training {metric_name}: {train_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)"""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        X = self.validate_input(X)
        return self.model.predict_proba(X)
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get weights or importance of each base model"""
        weights_dict = {}
        
        if self.ensemble_type == 'voting' and self.weights is not None:
            for (name, _), weight in zip(self.base_models, self.weights):
                weights_dict[name] = weight
        
        elif self.ensemble_type == 'stacking' and hasattr(self.meta_learner, 'coef_'):
            # Get meta-learner coefficients
            coefs = self.meta_learner.coef_.flatten()
            for i, (name, _) in enumerate(self.base_models):
                if i < len(coefs):
                    weights_dict[name] = abs(coefs[i])
        
        return weights_dict


class BlendingEnsemble:
    """Custom blending ensemble implementation"""
    
    def __init__(self, base_models: List[Tuple[str, Any]],
                 meta_learner: Any,
                 task_type: str = 'classification',
                 blend_features: bool = True):
        """
        Initialize blending ensemble
        
        Args:
            base_models: List of (name, model) tuples
            meta_learner: Model to blend predictions
            task_type: 'classification' or 'regression'
            blend_features: Whether to include original features in blending
        """
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.task_type = task_type
        self.blend_features = blend_features
        self.is_fitted = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        """
        Fit blending ensemble
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features for blending
            y_val: Validation targets
        """
        # Train base models on training data
        logger.info("Training base models...")
        for name, model in self.base_models:
            model.fit(X_train, y_train)
            logger.info(f"  Trained {name}")
        
        # Generate blend features
        blend_features = self._create_blend_features(X_val)
        
        # Train meta-learner on blend features
        logger.info("Training meta-learner on blend features...")
        self.meta_learner.fit(blend_features, y_val)
        
        self.is_fitted = True
        logger.info("Blending ensemble training complete")
    
    def _create_blend_features(self, X: np.ndarray) -> np.ndarray:
        """Create features for blending"""
        blend_predictions = []
        
        # Get predictions from each base model
        for name, model in self.base_models:
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                # Use probability predictions
                pred = model.predict_proba(X)[:, 1]
            else:
                # Use regular predictions
                pred = model.predict(X)
            
            blend_predictions.append(pred)
        
        # Stack predictions
        blend_array = np.column_stack(blend_predictions)
        
        # Optionally include original features
        if self.blend_features:
            blend_array = np.hstack([blend_array, X])
        
        return blend_array
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make blended predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        blend_features = self._create_blend_features(X)
        return self.meta_learner.predict(blend_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)"""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        blend_features = self._create_blend_features(X)
        return self.meta_learner.predict_proba(blend_features)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model score"""
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, predictions)
        else:
            from sklearn.metrics import r2_score
            return r2_score(y, predictions)


class DynamicEnsemble(BaseModel):
    """Dynamic ensemble that weights models based on recent performance"""
    
    def __init__(self, base_models: List[Tuple[str, Any]],
                 task_type: str = 'classification',
                 window_size: int = 100,
                 update_frequency: int = 10):
        """
        Initialize dynamic ensemble
        
        Args:
            base_models: List of (name, model) tuples
            task_type: 'classification' or 'regression'
            window_size: Number of recent predictions to consider
            update_frequency: How often to update weights
        """
        super().__init__(model_name="DynamicEnsemble")
        
        self.base_models = base_models
        self.task_type = task_type
        self.window_size = window_size
        self.update_frequency = update_frequency
        
        # Initialize tracking
        self.prediction_history = {name: [] for name, _ in base_models}
        self.performance_history = {name: [] for name, _ in base_models}
        self.current_weights = {name: 1.0 / len(base_models) for name, _ in base_models}
        self.prediction_count = 0
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all base models"""
        for name, model in self.base_models:
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Initialize weights based on cross-validation performance
            cv_scores = cross_val_predict(model, X_train, y_train, cv=5)
            if self.task_type == 'classification':
                initial_score = (cv_scores == y_train).mean()
            else:
                initial_score = 1 - np.mean((cv_scores - y_train) ** 2) / np.var(y_train)
            
            self.current_weights[name] = max(0.1, initial_score)
        
        # Normalize weights
        self._normalize_weights()
    
    def predict(self, X: np.ndarray, update_weights: bool = True) -> np.ndarray:
        """Make weighted ensemble predictions"""
        X = self.validate_input(X)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.base_models:
            predictions[name] = model.predict(X)
        
        # Combine predictions using current weights
        weighted_pred = np.zeros_like(predictions[self.base_models[0][0]])
        
        for name in predictions:
            weighted_pred += self.current_weights[name] * predictions[name]
        
        # Store predictions for weight updates
        if update_weights:
            for name in predictions:
                self.prediction_history[name].extend(predictions[name])
            self.prediction_count += len(X)
        
        # Update weights if needed
        if update_weights and self.prediction_count % self.update_frequency == 0:
            self._update_weights()
        
        return weighted_pred
    
    def update_performance(self, y_true: np.ndarray):
        """Update performance history with true labels"""
        # Calculate performance for recent predictions
        start_idx = -len(y_true)
        
        for name in self.prediction_history:
            recent_preds = self.prediction_history[name][start_idx:]
            
            if self.task_type == 'classification':
                score = (recent_preds == y_true).mean()
            else:
                score = 1 - np.mean((recent_preds - y_true) ** 2) / np.var(y_true)
            
            self.performance_history[name].append(score)
    
    def _update_weights(self):
        """Update model weights based on recent performance"""
        if not any(self.performance_history[name] for name in self.performance_history):
            return
        
        # Calculate recent performance
        for name in self.current_weights:
            if self.performance_history[name]:
                # Use exponentially weighted average
                recent_scores = self.performance_history[name][-self.window_size:]
                weights = np.exp(np.linspace(-1, 0, len(recent_scores)))
                weights /= weights.sum()
                
                weighted_score = np.average(recent_scores, weights=weights)
                self.current_weights[name] = max(0.1, weighted_score)
        
        self._normalize_weights()
        
        logger.info("Updated ensemble weights:")
        for name, weight in self.current_weights.items():
            logger.info(f"  {name}: {weight:.3f}")
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total = sum(self.current_weights.values())
        for name in self.current_weights:
            self.current_weights[name] /= total


class HierarchicalEnsemble(BaseModel):
    """Hierarchical ensemble for multi-stage predictions"""
    
    def __init__(self, stage_models: Dict[str, List[Tuple[str, Any]]],
                 stage_dependencies: Dict[str, List[str]]):
        """
        Initialize hierarchical ensemble
        
        Args:
            stage_models: Dictionary of stage_name -> list of (name, model) tuples
            stage_dependencies: Dictionary of stage_name -> list of dependent stages
        """
        super().__init__(model_name="HierarchicalEnsemble")
        
        self.stage_models = stage_models
        self.stage_dependencies = stage_dependencies
        self.stage_predictions = {}
        
        # Validate dependencies
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Ensure no circular dependencies"""
        visited = set()
        
        def has_cycle(stage, path):
            if stage in path:
                return True
            if stage in visited:
                return False
            
            visited.add(stage)
            path.add(stage)
            
            for dep in self.stage_dependencies.get(stage, []):
                if has_cycle(dep, path):
                    return True
            
            path.remove(stage)
            return False
        
        for stage in self.stage_models:
            if has_cycle(stage, set()):
                raise ValueError(f"Circular dependency detected involving {stage}")
    
    def train(self, X_train: np.ndarray, y_train: Dict[str, np.ndarray]):
        """
        Train hierarchical models
        
        Args:
            X_train: Training features
            y_train: Dictionary of stage_name -> targets
        """
        # Get execution order
        execution_order = self._get_execution_order()
        
        for stage in execution_order:
            logger.info(f"Training stage: {stage}")
            
            # Prepare features for this stage
            stage_features = self._prepare_stage_features(X_train, stage)
            
            # Train models in this stage
            for name, model in self.stage_models[stage]:
                model.fit(stage_features, y_train[stage])
                logger.info(f"  Trained {name}")
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make hierarchical predictions"""
        X = self.validate_input(X)
        
        execution_order = self._get_execution_order()
        self.stage_predictions = {}
        
        for stage in execution_order:
            # Prepare features including predictions from dependencies
            stage_features = self._prepare_stage_features(X, stage)
            
            # Get predictions from all models in this stage
            stage_preds = []
            for name, model in self.stage_models[stage]:
                pred = model.predict(stage_features)
                stage_preds.append(pred)
            
            # Average predictions (or use more sophisticated combination)
            self.stage_predictions[stage] = np.mean(stage_preds, axis=0)
        
        return self.stage_predictions
    
    def _get_execution_order(self) -> List[str]:
        """Determine order to execute stages"""
        order = []
        visited = set()
        
        def visit(stage):
            if stage in visited:
                return
            
            visited.add(stage)
            
            # Visit dependencies first
            for dep in self.stage_dependencies.get(stage, []):
                visit(dep)
            
            order.append(stage)
        
        for stage in self.stage_models:
            visit(stage)
        
        return order
    
    def _prepare_stage_features(self, X: np.ndarray, stage: str) -> np.ndarray:
        """Prepare features for a stage including dependency predictions"""
        features = [X]
        
        # Add predictions from dependent stages
        for dep_stage in self.stage_dependencies.get(stage, []):
            if dep_stage in self.stage_predictions:
                dep_pred = self.stage_predictions[dep_stage]
                if dep_pred.ndim == 1:
                    dep_pred = dep_pred.reshape(-1, 1)
                features.append(dep_pred)
        
        return np.hstack(features) if len(features) > 1 else features[0]