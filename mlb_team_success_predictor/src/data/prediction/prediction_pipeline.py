"""
Prediction pipelines for production use
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import logging
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from .predictor import (
    MLBPredictor, DivisionWinnerPredictor, WinsPredictor,
    MilestonePredictor, EnsemblePredictor, create_predictor
)
from ..data_preprocessor import DataPreprocessor
from ..feature_engineering import FeatureEngineer
from ..utils.config import get_config, SAVED_MODELS_PATH
from ..utils.helpers import timer, save_json, ensure_dir

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Complete prediction pipeline with preprocessing"""
    
    def __init__(self, predictor_type: str = 'ensemble',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize prediction pipeline
        
        Args:
            predictor_type: Type of predictor to use
            config: Configuration dictionary
        """
        self.predictor_type = predictor_type
        self.config = config or get_config()
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.predictor = create_predictor(predictor_type)
        
        # Pipeline state
        self.is_ready = False
        self._feature_cache = {}
        
    def initialize(self):
        """Initialize pipeline components"""
        logger.info("Initializing prediction pipeline...")
        
        # Load models
        if hasattr(self.predictor, 'load_all_models'):
            self.predictor.load_all_models()
        else:
            self.predictor.load_model()
            self.predictor.load_scaler()
        
        self.is_ready = True
        logger.info("Pipeline initialized successfully")
    
    def preprocess_input(self, raw_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess raw input data
        
        Args:
            raw_data: Raw input data
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert to DataFrame if necessary
        if isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        else:
            df = raw_data.copy()
        
        # Apply preprocessing
        df = self.preprocessor.preprocess(df)
        
        # Engineer features
        df = self.feature_engineer.engineer_features(df)
        
        return df
    
    def predict(self, raw_data: Union[Dict, pd.DataFrame],
                return_features: bool = False) -> Dict[str, Any]:
        """
        Make predictions from raw data
        
        Args:
            raw_data: Raw input data
            return_features: Whether to return engineered features
            
        Returns:
            Prediction results
        """
        if not self.is_ready:
            self.initialize()
        
        with timer("Complete prediction pipeline"):
            # Preprocess data
            processed_data = self.preprocess_input(raw_data)
            
            # Make predictions
            if isinstance(self.predictor, EnsemblePredictor):
                results = self.predictor.create_comprehensive_report(processed_data)
            elif isinstance(self.predictor, DivisionWinnerPredictor):
                results = self.predictor.predict_with_confidence(processed_data)
            elif isinstance(self.predictor, WinsPredictor):
                results = self.predictor.predict_with_bounds(processed_data)
            elif isinstance(self.predictor, MilestonePredictor):
                results = self.predictor.predict_milestones(processed_data)
            else:
                predictions = self.predictor.predict(processed_data)
                results = {'predictions': predictions.tolist()}
            
            # Add metadata
            results['metadata'] = {
                'predictor_type': self.predictor_type,
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(processed_data)
            }
            
            # Optionally return features
            if return_features:
                results['features'] = processed_data.to_dict('records')
        
        return results
    
    def validate_input(self, raw_data: Union[Dict, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validate input data
        
        Args:
            raw_data: Input data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Convert to DataFrame
        if isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        else:
            df = raw_data
        
        # Check required columns
        required_base_columns = ['wins', 'losses', 'runs_scored', 'runs_allowed']
        missing_columns = set(required_base_columns) - set(df.columns)
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        numeric_columns = ['wins', 'losses', 'runs_scored', 'runs_allowed']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric")
        
        # Check value ranges
        if 'wins' in df.columns and 'losses' in df.columns:
            total_games = df['wins'] + df['losses']
            if (total_games > 163).any():
                errors.append("Total games (wins + losses) exceeds maximum")
        
        is_valid = len(errors) == 0
        
        return is_valid, errors


class BatchPredictionPipeline(PredictionPipeline):
    """Pipeline for batch predictions"""
    
    def __init__(self, predictor_type: str = 'ensemble',
                 batch_size: int = 1000,
                 n_jobs: int = -1):
        """
        Initialize batch prediction pipeline
        
        Args:
            predictor_type: Type of predictor
            batch_size: Size of batches
            n_jobs: Number of parallel jobs
        """
        super().__init__(predictor_type)
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs > 0 else None
        
    def predict_batch(self, data: pd.DataFrame,
                     progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Make predictions on batch data
        
        Args:
            data: Input DataFrame
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_ready:
            self.initialize()
        
        n_samples = len(data)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {n_samples} samples in {n_batches} batches")
        
        results_list = []
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            # Get batch
            batch_data = data.iloc[start_idx:end_idx]
            
            # Process batch
            batch_results = self.predict(batch_data)
            
            # Extract predictions based on predictor type
            if self.predictor_type == 'division_winner':
                predictions = batch_results.get('predictions', [])
                probabilities = [p[1] for p in batch_results.get('probabilities', [])]
                
                batch_df = batch_data.copy()
                batch_df['division_winner_prediction'] = predictions
                batch_df['division_winner_probability'] = probabilities
                
            elif self.predictor_type == 'wins':
                predictions = batch_results.get('predictions', [])
                
                batch_df = batch_data.copy()
                batch_df['projected_wins'] = predictions
                
            else:
                # Generic handling
                batch_df = batch_data.copy()
                if 'predictions' in batch_results:
                    batch_df['prediction'] = batch_results['predictions']
            
            results_list.append(batch_df)
            
            # Progress callback
            if progress_callback:
                progress = (i + 1) / n_batches
                progress_callback(progress)
        
        # Combine results
        results_df = pd.concat(results_list, ignore_index=True)
        
        logger.info("Batch prediction complete")
        
        return results_df
    
    async def predict_batch_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Asynchronous batch prediction
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with predictions
        """
        n_samples = len(data)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        async def process_batch(batch_idx: int) -> pd.DataFrame:
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
            batch_data = data.iloc[start_idx:end_idx]
            
            # Run prediction in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor, self.predict, batch_data
                )
            
            # Process results
            batch_df = batch_data.copy()
            if 'predictions' in result:
                batch_df['prediction'] = result['predictions']
            
            return batch_df
        
        # Process all batches concurrently
        tasks = [process_batch(i) for i in range(n_batches)]
        batch_results = await asyncio.gather(*tasks)
        
        # Combine results
        results_df = pd.concat(batch_results, ignore_index=True)
        
        return results_df
    
    def save_predictions(self, predictions_df: pd.DataFrame,
                        output_path: Path,
                        format: str = 'csv'):
        """
        Save prediction results
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Output file path
            format: Output format ('csv', 'json', 'parquet')
        """
        ensure_dir(output_path.parent)
        
        if format == 'csv':
            predictions_df.to_csv(output_path, index=False)
        elif format == 'json':
            predictions_df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            predictions_df.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Predictions saved to {output_path}")


class RealTimePredictor:
    """Real-time prediction service"""
    
    def __init__(self, pipeline: PredictionPipeline,
                 cache_size: int = 1000,
                 cache_ttl: int = 3600):
        """
        Initialize real-time predictor
        
        Args:
            pipeline: Prediction pipeline
            cache_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
        """
        self.pipeline = pipeline
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        
        # Initialize cache
        self._prediction_cache = {}
        self._cache_timestamps = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_prediction_time': 0,
            'error_count': 0
        }
    
    def _get_cache_key(self, data: Dict) -> str:
        """Generate cache key from input data"""
        # Sort keys for consistency
        sorted_data = {k: data[k] for k in sorted(data.keys())}
        return json.dumps(sorted_data, sort_keys=True)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached prediction is still valid"""
        if key not in self._cache_timestamps:
            return False
        
        age = datetime.now().timestamp() - self._cache_timestamps[key]
        return age < self.cache_ttl
    
    def _update_cache(self, key: str, result: Dict):
        """Update prediction cache"""
        # Remove oldest entries if cache is full
        if len(self._prediction_cache) >= self.cache_size:
            oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
            del self._prediction_cache[oldest_key]
            del self._cache_timestamps[oldest_key]
        
        self._prediction_cache[key] = result
        self._cache_timestamps[key] = datetime.now().timestamp()
    
    def predict(self, data: Dict, use_cache: bool = True) -> Dict[str, Any]:
        """
        Make real-time prediction
        
        Args:
            data: Input data dictionary
            use_cache: Whether to use caching
            
        Returns:
            Prediction results
        """
        start_time = datetime.now()
        self.performance_stats['total_predictions'] += 1
        
        try:
            # Check cache
            if use_cache:
                cache_key = self._get_cache_key(data)
                
                if cache_key in self._prediction_cache and self._is_cache_valid(cache_key):
                    self.performance_stats['cache_hits'] += 1
                    result = self._prediction_cache[cache_key].copy()
                    result['from_cache'] = True
                    return result
                else:
                    self.performance_stats['cache_misses'] += 1
            
            # Validate input
            is_valid, errors = self.pipeline.validate_input(data)
            if not is_valid:
                return {
                    'error': 'Invalid input',
                    'errors': errors,
                    'status': 'failed'
                }
            
            # Make prediction
            result = self.pipeline.predict(data)
            result['from_cache'] = False
            
            # Update cache
            if use_cache:
                self._update_cache(cache_key, result)
            
            # Update performance stats
            prediction_time = (datetime.now() - start_time).total_seconds()
            n = self.performance_stats['total_predictions']
            self.performance_stats['avg_prediction_time'] = (
                (self.performance_stats['avg_prediction_time'] * (n - 1) + prediction_time) / n
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            self.performance_stats['error_count'] += 1
            
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        
        # Calculate cache hit rate
        total_cached_requests = stats['cache_hits'] + stats['cache_misses']
        if total_cached_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cached_requests
        else:
            stats['cache_hit_rate'] = 0
        
        # Add cache info
        stats['cache_size'] = len(self._prediction_cache)
        stats['cache_capacity'] = self.cache_size
        
        return stats
    
    def clear_cache(self):
        """Clear prediction cache"""
        self._prediction_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Prediction cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'pipeline_ready': self.pipeline.is_ready,
            'cache_enabled': True,
            'performance': self.get_performance_stats()
        }
        
        # Check if pipeline is working
        try:
            test_data = {
                'wins': 81,
                'losses': 81,
                'runs_scored': 700,
                'runs_allowed': 700,
                'run_differential': 0,
                'home_win_pct': 0.5,
                'away_win_pct': 0.5
            }
            
            result = self.predict(test_data, use_cache=False)
            if 'error' in result:
                health['status'] = 'unhealthy'
                health['error'] = result['error']
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health


# Factory function
def create_prediction_pipeline(pipeline_type: str = 'standard',
                              predictor_type: str = 'ensemble',
                              **kwargs) -> Union[PredictionPipeline, BatchPredictionPipeline, RealTimePredictor]:
    """
    Create a prediction pipeline
    
    Args:
        pipeline_type: Type of pipeline ('standard', 'batch', 'realtime')
        predictor_type: Type of predictor
        **kwargs: Additional arguments
        
    Returns:
        Pipeline instance
    """
    if pipeline_type == 'standard':
        return PredictionPipeline(predictor_type, **kwargs)
    
    elif pipeline_type == 'batch':
        batch_size = kwargs.pop('batch_size', 1000)
        n_jobs = kwargs.pop('n_jobs', -1)
        pipeline = BatchPredictionPipeline(predictor_type, batch_size, n_jobs)
        return pipeline
    
    elif pipeline_type == 'realtime':
        base_pipeline = PredictionPipeline(predictor_type, **kwargs)
        cache_size = kwargs.pop('cache_size', 1000)
        cache_ttl = kwargs.pop('cache_ttl', 3600)
        return RealTimePredictor(base_pipeline, cache_size, cache_ttl)
    
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


# Example service class for API integration
class PredictionService:
    """High-level prediction service for API integration"""
    
    def __init__(self):
        """Initialize prediction service"""
        # Create predictors for different tasks
        self.predictors = {
            'division_winner': create_prediction_pipeline('realtime', 'division_winner'),
            'wins': create_prediction_pipeline('realtime', 'wins'),
            'milestones': create_prediction_pipeline('realtime', 'milestone'),
            'comprehensive': create_prediction_pipeline('realtime', 'ensemble')
        }
        
        # Initialize all pipelines
        for predictor in self.predictors.values():
            predictor.pipeline.initialize()
    
    def predict_division_winner(self, team_stats: Dict) -> Dict[str, Any]:
        """Predict division winner probability"""
        return self.predictors['division_winner'].predict(team_stats)
    
    def predict_wins(self, team_stats: Dict) -> Dict[str, Any]:
        """Predict season win total"""
        return self.predictors['wins'].predict(team_stats)
    
    def predict_milestones(self, team_stats: Dict) -> Dict[str, Any]:
        """Predict milestone achievements"""
        return self.predictors['milestones'].predict(team_stats)
    
    def predict_all(self, team_stats: Dict) -> Dict[str, Any]:
        """Get comprehensive predictions"""
        return self.predictors['comprehensive'].predict(team_stats)
    
    def predict_season(self, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Predict full season for all teams"""
        batch_pipeline = create_prediction_pipeline('batch', 'ensemble')
        return batch_pipeline.predict_batch(teams_df)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all services"""
        health = {
            'overall_status': 'healthy',
            'services': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for name, predictor in self.predictors.items():
            service_health = predictor.health_check()
            health['services'][name] = service_health
            
            if service_health['status'] != 'healthy':
                health['overall_status'] = 'degraded'
        
        return health


if __name__ == "__main__":
    # Example usage
    # Create pipeline
    pipeline = create_prediction_pipeline('standard', 'division_winner')
    
    # Sample team data
    team_data = {
        'team_name': 'Sample Team',
        'wins': 88,
        'losses': 74,
        'winning_percentage': 0.543,
        'run_differential': 50,
        'home_win_pct': 0.580,
        'away_win_pct': 0.506,
        'runs_per_game': 4.8,
        'runs_allowed_per_game': 4.5,
        'pythagorean_wins': 87,
        'prev_year_wins': 85
    }
    
    # Make prediction
    # result = pipeline.predict(team_data)
    # print(json.dumps(result, indent=2))
    
    # Batch prediction example
    teams_df = pd.DataFrame([team_data] * 100)  # Simulate 100 teams
    batch_pipeline = create_prediction_pipeline('batch', 'wins')
    # results_df = batch_pipeline.predict_batch(teams_df)
    
    # Real-time service example
    service = PredictionService()
    # health = service.get_health_status()
    # print(json.dumps(health, indent=2))