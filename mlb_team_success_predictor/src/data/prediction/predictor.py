"""
Main prediction classes for MLB models
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import json

from ..models.classification_models import DivisionWinnerClassifier, PlayoffClassifier
from ..models.regression_models import WinsRegressor, RunProductionRegressor
from ..models.milestone_predictor import MilestonePredictor
from ..models.ensemble_models import MLBEnsembleModel
from ..utils.config import SAVED_MODELS_PATH, SCALERS_PATH
from ..utils.helpers import timer, ensure_dir

logger = logging.getLogger(__name__)


class MLBPredictor:
    """Base predictor class for MLB models"""
    
    def __init__(self, model_type: str, model_path: Optional[Path] = None,
                 scaler_path: Optional[Path] = None):
        """
        Initialize predictor
        
        Args:
            model_type: Type of model ('classification', 'regression', etc.)
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        self.model_type = model_type
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.feature_names = []
        
    def load_model(self, model_path: Optional[Path] = None):
        """Load model from disk"""
        path = model_path or self.model_path
        
        if path is None:
            raise ValueError("No model path provided")
        
        try:
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            
            # Extract feature names if available
            if hasattr(self.model, 'feature_names'):
                self.feature_names = self.model.feature_names
                
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_scaler(self, scaler_path: Optional[Path] = None):
        """Load scaler from disk"""
        path = scaler_path or self.scaler_path
        
        if path is None:
            logger.warning("No scaler path provided, predictions will use raw features")
            return
        
        try:
            self.scaler = joblib.load(path)
            logger.info(f"Scaler loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise
    
    def prepare_features(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Prepare features for prediction
        
        Args:
            data: Input data (dict, DataFrame, or array)
            
        Returns:
            Prepared feature array
        """
        # Convert to DataFrame if necessary
        if isinstance(data, dict):
            # Single prediction
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, np.ndarray):
            if len(self.feature_names) > 0:
                df = pd.DataFrame(data, columns=self.feature_names)
            else:
                # Use array directly
                return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Select and order features
        if self.feature_names:
            try:
                features = df[self.feature_names].values
            except KeyError as e:
                missing = set(self.feature_names) - set(df.columns)
                raise ValueError(f"Missing required features: {missing}")
        else:
            features = df.values
        
        # Apply scaling if available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features
    
    def predict(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        features = self.prepare_features(data)
        
        with timer("Prediction", logger.info):
            predictions = self.model.predict(features)
        
        return predictions
    
    def predict_proba(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get prediction probabilities (classification only)"""
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        features = self.prepare_features(data)
        
        with timer("Probability prediction", logger.info):
            probabilities = self.model.predict_proba(features)
        
        return probabilities


class DivisionWinnerPredictor(MLBPredictor):
    """Predictor for division winner classification"""
    
    def __init__(self, model_path: Optional[Path] = None,
                 scaler_path: Optional[Path] = None):
        """Initialize division winner predictor"""
        super().__init__('classification', model_path, scaler_path)
        
        # Default paths
        if model_path is None:
            self.model_path = SAVED_MODELS_PATH / 'division_winner_best_model.pkl'
        if scaler_path is None:
            self.scaler_path = SCALERS_PATH / 'division_winner_scaler.pkl'
    
    def predict_with_confidence(self, data: Union[Dict, pd.DataFrame, np.ndarray],
                               confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Make predictions with confidence levels
        
        Args:
            data: Input data
            confidence_threshold: Threshold for high confidence
            
        Returns:
            Dictionary with predictions and confidence
        """
        # Get predictions and probabilities
        predictions = self.predict(data)
        probabilities = self.predict_proba(data)
        
        # Calculate confidence
        if probabilities.shape[1] == 2:
            # Binary classification
            prob_positive = probabilities[:, 1]
            confidence = np.abs(prob_positive - 0.5) * 2
        else:
            # Multi-class
            confidence = np.max(probabilities, axis=1)
        
        # Determine confidence levels
        confidence_levels = np.where(
            confidence >= confidence_threshold, 'high',
            np.where(confidence >= 0.4, 'medium', 'low')
        )
        
        # Format results
        results = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'confidence_scores': confidence.tolist(),
            'confidence_levels': confidence_levels.tolist()
        }
        
        # Add single prediction format if only one sample
        if len(predictions) == 1:
            results['prediction'] = int(predictions[0])
            results['probability'] = float(prob_positive[0]) if probabilities.shape[1] == 2 else probabilities[0].tolist()
            results['confidence'] = float(confidence[0])
            results['confidence_level'] = confidence_levels[0]
        
        return results
    
    def predict_division_winners(self, teams_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict division winners from team data
        
        Args:
            teams_data: DataFrame with team statistics
            
        Returns:
            DataFrame with predictions and rankings
        """
        # Make predictions
        predictions = self.predict_with_confidence(teams_data)
        
        # Add to dataframe
        results_df = teams_data.copy()
        results_df['division_winner_probability'] = [p[1] for p in predictions['probabilities']]
        results_df['confidence'] = predictions['confidence_scores']
        
        # Rank by division if division column exists
        if 'division' in results_df.columns:
            results_df['division_rank'] = results_df.groupby('division')['division_winner_probability'].rank(
                ascending=False, method='min'
            )
        
        # Sort by probability
        results_df = results_df.sort_values('division_winner_probability', ascending=False)
        
        return results_df


class WinsPredictor(MLBPredictor):
    """Predictor for season win totals"""
    
    def __init__(self, model_path: Optional[Path] = None,
                 scaler_path: Optional[Path] = None):
        """Initialize wins predictor"""
        super().__init__('regression', model_path, scaler_path)
        
        # Default paths
        if model_path is None:
            self.model_path = SAVED_MODELS_PATH / 'wins_best_regressor.pkl'
        if scaler_path is None:
            self.scaler_path = SCALERS_PATH / 'wins_regressor_scaler.pkl'
    
    def predict_with_bounds(self, data: Union[Dict, pd.DataFrame, np.ndarray],
                           confidence_level: float = 0.95,
                           historical_std: float = 10.0) -> Dict[str, Any]:
        """
        Predict wins with confidence bounds
        
        Args:
            data: Input data
            confidence_level: Confidence level for bounds
            historical_std: Historical standard deviation of predictions
            
        Returns:
            Dictionary with predictions and bounds
        """
        # Get point predictions
        predictions = self.predict(data)
        
        # Calculate bounds (simplified - should use proper prediction intervals)
        z_score = 1.96 if confidence_level == 0.95 else 2.58
        lower_bounds = predictions - z_score * historical_std
        upper_bounds = predictions + z_score * historical_std
        
        # Clip to valid range
        lower_bounds = np.clip(lower_bounds, 0, 162)
        upper_bounds = np.clip(upper_bounds, 0, 162)
        predictions = np.clip(predictions, 0, 162)
        
        # Format results
        results = {
            'predictions': predictions.tolist(),
            'lower_bounds': lower_bounds.tolist(),
            'upper_bounds': upper_bounds.tolist(),
            'confidence_level': confidence_level
        }
        
        # Add single prediction format if only one sample
        if len(predictions) == 1:
            results['prediction'] = int(predictions[0])
            results['prediction_interval'] = [int(lower_bounds[0]), int(upper_bounds[0])]
        
        return results
    
    def predict_season_projections(self, teams_data: pd.DataFrame,
                                  games_played: Optional[int] = None) -> pd.DataFrame:
        """
        Project full season wins
        
        Args:
            teams_data: DataFrame with current team statistics
            games_played: Number of games played so far
            
        Returns:
            DataFrame with projections
        """
        # Make predictions
        predictions = self.predict_with_bounds(teams_data)
        
        # Add to dataframe
        results_df = teams_data.copy()
        results_df['projected_wins'] = predictions['predictions']
        results_df['win_projection_low'] = predictions['lower_bounds']
        results_df['win_projection_high'] = predictions['upper_bounds']
        
        # If mid-season, blend with current pace
        if games_played and games_played > 0 and 'wins' in results_df.columns:
            current_pace = (results_df['wins'] / games_played) * 162
            weight = games_played / 162
            
            results_df['blended_projection'] = (
                weight * current_pace + (1 - weight) * results_df['projected_wins']
            )
        
        # Add win categories
        results_df['projection_category'] = pd.cut(
            results_df['projected_wins'],
            bins=[0, 65, 75, 85, 95, 105, 163],
            labels=['Terrible', 'Poor', 'Below Average', 'Good', 'Excellent', 'Elite']
        )
        
        # Sort by projected wins
        results_df = results_df.sort_values('projected_wins', ascending=False)
        
        return results_df


class MilestonePredictor(MLBPredictor):
    """Predictor for milestone achievements"""
    
    def __init__(self, model_path: Optional[Path] = None,
                 scaler_path: Optional[Path] = None):
        """Initialize milestone predictor"""
        super().__init__('milestone', model_path, scaler_path)
        
        # Default paths
        if model_path is None:
            self.model_path = SAVED_MODELS_PATH / 'milestone_predictor.pkl'
        if scaler_path is None:
            self.scaler_path = SCALERS_PATH / 'milestone_scaler.pkl'
        
        # Milestone definitions
        self.milestone_names = [
            'achieved_90_wins',
            'achieved_100_wins',
            'scored_800_runs',
            'made_playoffs'
        ]
    
    def predict_milestones(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Predict milestone achievements
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with milestone predictions
        """
        if not self.is_loaded:
            self.load_model()
            self.load_scaler()
        
        features = self.prepare_features(data)
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            # Get probabilities for each milestone
            probabilities = []
            for estimator in self.model.estimators_:
                proba = estimator.predict_proba(features)[:, 1]
                probabilities.append(proba)
            probabilities = np.column_stack(probabilities)
        else:
            # Binary predictions
            predictions = self.model.predict(features)
            probabilities = predictions
        
        # Format results
        results = {
            'milestones': {}
        }
        
        for i, milestone in enumerate(self.milestone_names):
            if probabilities.ndim > 1:
                milestone_probs = probabilities[:, i]
            else:
                milestone_probs = probabilities if len(self.milestone_names) == 1 else [0.5]
            
            results['milestones'][milestone] = {
                'probabilities': milestone_probs.tolist(),
                'predictions': (milestone_probs >= 0.5).astype(int).tolist()
            }
            
            # Add summary for single prediction
            if len(milestone_probs) == 1:
                results['milestones'][milestone]['probability'] = float(milestone_probs[0])
                results['milestones'][milestone]['prediction'] = bool(milestone_probs[0] >= 0.5)
        
        return results
    
    def create_milestone_report(self, teams_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive milestone report for teams
        
        Args:
            teams_data: DataFrame with team statistics
            
        Returns:
            DataFrame with milestone predictions
        """
        # Get predictions
        predictions = self.predict_milestones(teams_data)
        
        # Create report dataframe
        report_df = teams_data[['team_name']].copy() if 'team_name' in teams_data else pd.DataFrame()
        
        # Add milestone probabilities
        for milestone in self.milestone_names:
            probs = predictions['milestones'][milestone]['probabilities']
            report_df[f'{milestone}_probability'] = probs
        
        # Add summary columns
        milestone_probs = report_df[[col for col in report_df.columns if '_probability' in col]]
        report_df['avg_milestone_probability'] = milestone_probs.mean(axis=1)
        report_df['likely_milestones'] = milestone_probs.apply(
            lambda x: sum(x >= 0.5), axis=1
        )
        
        # Sort by average probability
        report_df = report_df.sort_values('avg_milestone_probability', ascending=False)
        
        return report_df


class EnsemblePredictor(MLBPredictor):
    """Ensemble predictor combining multiple models"""
    
    def __init__(self, predictors: Dict[str, MLBPredictor]):
        """
        Initialize ensemble predictor
        
        Args:
            predictors: Dictionary of name -> predictor
        """
        super().__init__('ensemble')
        self.predictors = predictors
        self.is_loaded = all(p.is_loaded for p in predictors.values())
    
    def load_all_models(self):
        """Load all component models"""
        for name, predictor in self.predictors.items():
            if not predictor.is_loaded:
                logger.info(f"Loading {name} model...")
                predictor.load_model()
                predictor.load_scaler()
        
        self.is_loaded = True
    
    def predict_all(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Get predictions from all models
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with all predictions
        """
        if not self.is_loaded:
            self.load_all_models()
        
        results = {}
        
        for name, predictor in self.predictors.items():
            try:
                if isinstance(predictor, DivisionWinnerPredictor):
                    results[name] = predictor.predict_with_confidence(data)
                elif isinstance(predictor, WinsPredictor):
                    results[name] = predictor.predict_with_bounds(data)
                elif isinstance(predictor, MilestonePredictor):
                    results[name] = predictor.predict_milestones(data)
                else:
                    results[name] = {'predictions': predictor.predict(data).tolist()}
                    
            except Exception as e:
                logger.error(f"Error in {name} prediction: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def create_comprehensive_report(self, team_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """
        Create comprehensive prediction report
        
        Args:
            team_data: Team statistics
            
        Returns:
            Comprehensive prediction report
        """
        # Get all predictions
        all_predictions = self.predict_all(team_data)
        
        # Create summary
        report = {
            'timestamp': datetime.now().isoformat(),
            'predictions': all_predictions,
            'summary': {}
        }
        
        # Extract key predictions
        if 'division_winner' in all_predictions and 'probability' in all_predictions['division_winner']:
            report['summary']['division_winner_probability'] = all_predictions['division_winner']['probability']
        
        if 'wins' in all_predictions and 'prediction' in all_predictions['wins']:
            report['summary']['projected_wins'] = all_predictions['wins']['prediction']
            report['summary']['win_range'] = all_predictions['wins'].get('prediction_interval', [])
        
        if 'milestones' in all_predictions:
            milestone_summary = {}
            for milestone, data in all_predictions['milestones']['milestones'].items():
                if 'probability' in data:
                    milestone_summary[milestone] = data['probability']
            report['summary']['milestone_probabilities'] = milestone_summary
        
        return report


# Factory function
def create_predictor(predictor_type: str, **kwargs) -> MLBPredictor:
    """
    Create a predictor instance
    
    Args:
        predictor_type: Type of predictor ('division_winner', 'wins', 'milestone', 'ensemble')
        **kwargs: Additional arguments for predictor
        
    Returns:
        Predictor instance
    """
    predictors = {
        'division_winner': DivisionWinnerPredictor,
        'wins': WinsPredictor,
        'milestone': MilestonePredictor
    }
    
    if predictor_type == 'ensemble':
        # Create ensemble with all predictors
        components = {}
        for name, predictor_class in predictors.items():
            components[name] = predictor_class(**kwargs)
        return EnsemblePredictor(components)
    
    elif predictor_type in predictors:
        return predictors[predictor_type](**kwargs)
    
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


if __name__ == "__main__":
    # Example usage
    # Create sample team data
    team_data = {
        'wins': 85,
        'losses': 77,
        'winning_percentage': 0.525,
        'run_differential': 25,
        'home_win_pct': 0.550,
        'away_win_pct': 0.500,
        'runs_per_game': 4.5,
        'runs_allowed_per_game': 4.3,
        'pythagorean_wins': 84,
        'prev_year_wins': 82
    }
    
    # Create predictor
    predictor = create_predictor('division_winner')
    
    # Load model (would need actual model file)
    # predictor.load_model()
    # predictor.load_scaler()
    
    # Make prediction
    # result = predictor.predict_with_confidence(team_data)
    # print(f"Division winner probability: {result['probability']:.2%}")