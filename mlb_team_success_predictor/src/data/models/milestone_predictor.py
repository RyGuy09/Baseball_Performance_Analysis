"""
Multi-output models for predicting milestone achievements
"""

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Any, Optional, Union
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class MilestonePredictor(BaseModel):
    """Multi-output model for predicting various milestone achievements"""
    
    def __init__(self, model_type: str = 'random_forest',
                 milestone_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
                 use_chain: bool = False,
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize milestone predictor
        
        Args:
            model_type: Type of base model
            milestone_definitions: Dictionary defining milestones
            use_chain: Whether to use classifier chains
            model_params: Optional model parameters
        """
        super().__init__(model_name=f"MilestonePredictor_{model_type}")
        
        self.model_type = model_type
        self.use_chain = use_chain
        self.model_params = model_params or {}
        
        # Define default milestones
        self.milestone_definitions = milestone_definitions or {
            'achieved_90_wins': {
                'type': 'classification',
                'threshold': 90,
                'target': 'wins'
            },
            'achieved_100_wins': {
                'type': 'classification',
                'threshold': 100,
                'target': 'wins'
            },
            'scored_800_runs': {
                'type': 'classification',
                'threshold': 800,
                'target': 'runs_scored'
            },
            'elite_run_differential': {
                'type': 'classification',
                'threshold': 150,
                'target': 'run_differential'
            },
            'made_playoffs': {
                'type': 'classification',
                'custom': True
            }
        }
        
        self.milestone_names = list(self.milestone_definitions.keys())
        
        # Initialize model
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize multi-output model"""
        # Base estimator
        if self.model_type == 'random_forest':
            base_estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                **self.model_params
            )
        elif self.model_type == 'xgboost':
            base_estimator = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                **self.model_params
            )
        elif self.model_type == 'lightgbm':
            base_estimator = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                force_col_wise=True,
                **self.model_params
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Use classifier chains if requested (accounts for label correlations)
        if self.use_chain:
            from sklearn.multioutput import ClassifierChain
            return ClassifierChain(base_estimator, order='random', random_state=42)
        else:
            return MultiOutputClassifier(base_estimator)
    
    def prepare_milestone_targets(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare milestone target matrix from dataframe
        
        Args:
            df: DataFrame with raw statistics
            
        Returns:
            Binary matrix of milestone achievements
        """
        targets = []
        
        for milestone_name, definition in self.milestone_definitions.items():
            if definition.get('custom'):
                # Custom milestone logic
                if milestone_name == 'made_playoffs':
                    # Simplified playoff logic
                    target = (df['wins'] >= 90) | (df['games_behind'].isna())
                else:
                    raise ValueError(f"Unknown custom milestone: {milestone_name}")
            else:
                # Threshold-based milestone
                column = definition['target']
                threshold = definition['threshold']
                target = df[column] >= threshold
            
            targets.append(target.values)
        
        return np.column_stack(targets)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train milestone predictor
        
        Args:
            X_train: Training features
            y_train: Binary matrix of milestone achievements
        """
        logger.info(f"Training milestone predictor with {y_train.shape[1]} milestones...")
        
        # Validate target shape
        if y_train.shape[1] != len(self.milestone_names):
            raise ValueError(f"Expected {len(self.milestone_names)} milestone targets, "
                           f"got {y_train.shape[1]}")
        
        self.model.fit(X_train, y_train)
        
        # Log milestone achievement rates
        achievement_rates = y_train.mean(axis=0)
        for i, milestone in enumerate(self.milestone_names):
            logger.info(f"  {milestone}: {achievement_rates[i]:.1%} achievement rate")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict milestone achievements"""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Predict probabilities for each milestone
        
        Returns:
            List of probability arrays, one per milestone
        """
        X = self.validate_input(X)
        
        if hasattr(self.model, 'predict_proba'):
            # MultiOutputClassifier returns list of arrays
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            predictions = self.predict(X)
            # Convert to probability-like format
            return [np.column_stack([1 - pred, pred]) for pred in predictions.T]
    
    def predict_milestone_dict(self, X: np.ndarray) -> List[Dict[str, float]]:
        """
        Predict milestones and return as dictionary
        
        Args:
            X: Features for prediction
            
        Returns:
            List of dictionaries with milestone probabilities
        """
        probas = self.predict_proba(X)
        
        results = []
        for i in range(X.shape[0]):
            result = {}
            for j, milestone_name in enumerate(self.milestone_names):
                # Get probability of achieving milestone
                result[milestone_name] = float(probas[j][i, 1])
            results.append(result)
        
        return results
    
    def get_milestone_correlations(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Calculate correlations between milestones
        
        Args:
            X: Features (unused, for API consistency)
            y: Milestone achievement matrix
            
        Returns:
            Correlation matrix as DataFrame
        """
        milestone_df = pd.DataFrame(y, columns=self.milestone_names)
        return milestone_df.corr()
    
    def get_milestone_importance(self) -> pd.DataFrame:
        """Get feature importance for each milestone"""
        if not hasattr(self.model, 'estimators_'):
            logger.warning("Feature importance not available for this model type")
            return None
        
        importance_data = []
        
        for i, (milestone_name, estimator) in enumerate(zip(self.milestone_names, 
                                                           self.model.estimators_)):
            if hasattr(estimator, 'feature_importances_'):
                for j, importance in enumerate(estimator.feature_importances_):
                    importance_data.append({
                        'milestone': milestone_name,
                        'feature_index': j,
                        'feature': self.feature_names[j] if self.feature_names else f'feature_{j}',
                        'importance': importance
                    })
        
        importance_df = pd.DataFrame(importance_data)
        
        # Aggregate by feature across milestones
        avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        return avg_importance


class SeasonProjector(BaseModel):
    """Project full season statistics from partial season data"""
    
    def __init__(self, projection_method: str = 'weighted',
                 games_played_weight: float = 0.7):
        """
        Initialize season projector
        
        Args:
            projection_method: 'simple', 'weighted', or 'regression'
            games_played_weight: Weight for actual vs projected stats
        """
        super().__init__(model_name=f"SeasonProjector_{projection_method}")
        
        self.projection_method = projection_method
        self.games_played_weight = games_played_weight
        self.projection_model = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              games_played: np.ndarray, total_games: int = 162):
        """
        Train projection model
        
        Args:
            X_train: Partial season statistics
            y_train: Full season outcomes
            games_played: Number of games played for each sample
            total_games: Total games in season
        """
        self.total_games = total_games
        
        if self.projection_method == 'regression':
            # Train regression model for residuals
            # Calculate simple projections
            simple_projections = (X_train / games_played[:, np.newaxis]) * total_games
            
            # Calculate residuals
            residuals = y_train - simple_projections[:, 0]  # Assuming first feature is target
            
            # Train model to predict residuals
            from sklearn.ensemble import RandomForestRegressor
            self.projection_model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Features for residual prediction
            residual_features = np.column_stack([
                X_train,
                games_played,
                games_played / total_games,  # Completion percentage
                simple_projections
            ])
            
            self.projection_model.fit(residual_features, residuals)
    
    def predict(self, X: np.ndarray, games_played: np.ndarray) -> np.ndarray:
        """
        Project full season statistics
        
        Args:
            X: Current statistics
            games_played: Games played so far
            
        Returns:
            Projected full season statistics
        """
        X = self.validate_input(X)
        
        if self.projection_method == 'simple':
            # Simple linear projection
            return (X / games_played[:, np.newaxis]) * self.total_games
        
        elif self.projection_method == 'weighted':
            # Weight between current pace and historical average
            current_pace = (X / games_played[:, np.newaxis]) * self.total_games
            
            # For demonstration, assume league average (would be fitted)
            league_average = np.ones_like(X) * 81  # .500 record
            
            # Weight based on games played
            weight = np.minimum(games_played / self.total_games, 1.0) * self.games_played_weight
            weight = weight[:, np.newaxis]
            
            return weight * current_pace + (1 - weight) * league_average
        
        elif self.projection_method == 'regression':
            # Use trained model
            simple_projections = (X / games_played[:, np.newaxis]) * self.total_games
            
            residual_features = np.column_stack([
                X,
                games_played,
                games_played / self.total_games,
                simple_projections
            ])
            
            residuals = self.projection_model.predict(residual_features)
            
            return simple_projections[:, 0] + residuals
        
        else:
            raise ValueError(f"Unknown projection method: {self.projection_method}")