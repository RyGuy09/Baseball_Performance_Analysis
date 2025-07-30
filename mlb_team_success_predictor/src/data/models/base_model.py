"""
Base model class with common functionality for all MLB prediction models
"""

from abc import ABC, abstractmethod
import pickle
import joblib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all MLB prediction models"""
    
    def __init__(self, model_name: str = "BaseModel"):
        """
        Initialize base model
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_params = {}
        self.training_metadata = {}
        self.is_fitted = False
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        pass
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            feature_names: Optional[List[str]] = None, **kwargs):
        """
        Fit model with metadata tracking
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Names of features
            **kwargs: Additional parameters
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Store training metadata
        self.training_metadata = {
            'train_date': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'model_name': self.model_name,
            'model_class': self.__class__.__name__
        }
        
        # Train the model
        self.train(X_train, y_train, **kwargs)
        self.is_fitted = True
        
        logger.info(f"Model {self.model_name} trained on {len(X_train)} samples")
    
    def validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate input data before prediction
        
        Args:
            X: Input features
            
        Returns:
            Validated input array
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        # Check dimensions
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Check feature count
        if self.feature_names and X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"but got {X.shape[1]}"
            )
        
        # Check for NaN values
        if np.any(np.isnan(X)):
            logger.warning("Input contains NaN values")
        
        return X
    
    def save_model(self, filepath: Path, include_metadata: bool = True):
        """
        Save model to disk
        
        Args:
            filepath: Path to save the model
            include_metadata: Whether to save metadata alongside
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save main model
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
            
            # Save metadata if requested
            if include_metadata:
                metadata = {
                    'model_name': self.model_name,
                    'model_class': self.__class__.__name__,
                    'feature_names': self.feature_names,
                    'model_params': self.model_params,
                    'training_metadata': self.training_metadata,
                    'save_date': datetime.now().isoformat()
                }
                
                metadata_path = filepath.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Metadata saved to {metadata_path}")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: Path, load_metadata: bool = True):
        """
        Load model from disk
        
        Args:
            filepath: Path to the saved model
            load_metadata: Whether to load metadata
        """
        filepath = Path(filepath)
        
        try:
            # Load main model
            self.model = joblib.dump(filepath)
            logger.info(f"Model loaded from {filepath}")
            
            # Load metadata if requested
            if load_metadata:
                metadata_path = filepath.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.model_name = metadata.get('model_name', self.model_name)
                    self.feature_names = metadata.get('feature_names', [])
                    self.model_params = metadata.get('model_params', {})
                    self.training_metadata = metadata.get('training_metadata', {})
                    
                    logger.info("Model metadata loaded")
                else:
                    logger.warning(f"Metadata file not found: {metadata_path}")
            
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_scaler(self, filepath: Path):
        """Save feature scaler"""
        if self.scaler is not None:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.scaler, filepath)
            logger.info(f"Scaler saved to {filepath}")
        else:
            logger.warning("No scaler to save")
    
    def load_scaler(self, filepath: Path):
        """Load feature scaler"""
        filepath = Path(filepath)
        
        if filepath.exists():
            self.scaler = joblib.load(filepath)
            logger.info(f"Scaler loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata"""
        info = {
            'model_name': self.model_name,
            'model_class': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names[:10] if self.feature_names else [],
            'model_params': self.model_params,
            'training_metadata': self.training_metadata
        }
        
        # Add model-specific info if available
        if hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()
        
        return info
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return None
        
        # Check if model has feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': np.abs(self.model.coef_.flatten())
            }).sort_values('coefficient', ascending=False)
            
            return importance_df
        
        else:
            logger.info("Feature importance not available for this model type")
            return None
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        """
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            logger.warning("Cannot plot feature importance")
            return None
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=figsize)
        
        # Get top N features
        plot_df = importance_df.head(top_n).copy()
        
        # Create horizontal bar plot
        if 'importance' in plot_df.columns:
            sns.barplot(data=plot_df, x='importance', y='feature', palette='viridis')
            plt.xlabel('Feature Importance')
        else:
            sns.barplot(data=plot_df, x='coefficient', y='feature', palette='viridis')
            plt.xlabel('Feature Coefficient (Absolute Value)')
        
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features - {self.model_name}')
        plt.tight_layout()
        
        return plt.gcf()
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5,
                      scoring: str = 'accuracy') -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Targets
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist(),
            'scoring_metric': scoring,
            'n_folds': cv
        }
        
        logger.info(f"Cross-validation {scoring}: {results['mean_score']:.4f} "
                   f"(+/- {results['std_score']:.4f})")
        
        return results