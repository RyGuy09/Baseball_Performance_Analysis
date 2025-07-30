"""
Training pipeline for regression models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.data_loader import DataLoader
from ..data.data_preprocessor import DataPreprocessor
from ..data.feature_engineering import FeatureEngineer
from ..data.models.regression_models import WinsRegressor, RunProductionRegressor
from ..data.utils.config import (
    REGRESSION_FEATURES,
    SAVED_MODELS_PATH,
    SCALERS_PATH
)

logger = logging.getLogger(__name__)


class RegressorTrainer:
    """Comprehensive training pipeline for regression models"""
    
    def __init__(self, target_type: str = 'wins',
                 era_strategy: str = 'modern',
                 model_types: Optional[List[str]] = None):
        """
        Initialize regressor trainer
        
        Args:
            target_type: 'wins', 'runs_scored', or 'runs_allowed'
            era_strategy: 'modern', 'historical', or 'all'
            model_types: List of model types to train
        """
        self.target_type = target_type
        self.era_strategy = era_strategy
        self.model_types = model_types or ['random_forest', 'xgboost', 'lightgbm', 'elastic_net']
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor(era_strategy=era_strategy)
        self.feature_engineer = FeatureEngineer()
        
        # Storage
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = None
        
    def prepare_data(self, custom_features: Optional[List[str]] = None) -> Tuple[np.ndarray, ...]:
        """
        Load and prepare data for training
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, X_val, y_val)
        """
        logger.info(f"Preparing data for {self.target_type} regression...")
        
        # Load and process data
        df = self.data_loader.load_and_validate()
        df = self.preprocessor.preprocess(df)
        df = self.feature_engineer.engineer_features(df)
        
        # Select features and target
        features = custom_features or REGRESSION_FEATURES
        
        # Remove target from features if present
        if self.target_type in features:
            features = [f for f in features if f != self.target_type]
        
        # Clean data
        required_cols = features + [self.target_type]
        df_clean = df.dropna(subset=required_cols)
        
        logger.info(f"Clean data shape: {df_clean.shape}")
        
        # Prepare arrays
        X = df_clean[features].values
        y = df_clean[self.target_type].values
        
        # Create temporal split for time series nature of data
        # Sort by year and use recent years for test
        df_clean = df_clean.sort_values('year')
        
        # Use last 20% of years for test
        split_year = df_clean['year'].quantile(0.8)
        train_val_mask = df_clean['year'] < split_year
        test_mask = df_clean['year'] >= split_year
        
        X_train_val = X[train_val_mask]
        y_train_val = y[train_val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store metadata
        self.feature_names = features
        self.train_years = df_clean[train_val_mask]['year'].unique()
        self.test_years = df_clean[test_mask]['year'].unique()
        
        # Log statistics
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Target mean - Train: {y_train.mean():.2f}, Test: {y_test.mean():.2f}")
        logger.info(f"Target std - Train: {y_train.std():.2f}, Test: {y_test.std():.2f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_val_scaled, y_val
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray):
        """Train multiple regression models"""
        logger.info(f"Training {len(self.model_types)} regression models...")
        
        for model_type in self.model_types:
            logger.info(f"\nTraining {model_type}...")
            
            try:
                # Initialize model
                if self.target_type == 'wins':
                    model = WinsRegressor(model_type=model_type)
                else:
                    model = RunProductionRegressor(
                        target_type=self.target_type,
                        model_type=model_type
                    )
                
                # Set feature names
                model.feature_names = self.feature_names
                
                # Train
                start_time = datetime.now()
                model.train(X_train, y_train, X_val, y_val)
                train_time = (datetime.now() - start_time).total_seconds()
                
                # Store model
                self.models[model_type] = model
                
                # Evaluate
                val_metrics = self._evaluate_model(model, X_val, y_val)
                val_metrics['train_time'] = train_time
                
                self.results[model_type] = val_metrics
                
                logger.info(f"{model_type} - Val RMSE: {val_metrics['rmse']:.2f}, "
                           f"R²: {val_metrics['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                self.results[model_type] = {'error': str(e)}
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model"""
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mean_error': np.mean(y_pred - y),
            'std_error': np.std(y_pred - y),
            'median_absolute_error': np.median(np.abs(y_pred - y))
        }
        
        # Percentage within thresholds
        errors = np.abs(y_pred - y)
        if self.target_type == 'wins':
            metrics['within_5_wins'] = np.mean(errors <= 5)
            metrics['within_10_wins'] = np.mean(errors <= 10)
        else:  # runs
            metrics['within_25_runs'] = np.mean(errors <= 25)
            metrics['within_50_runs'] = np.mean(errors <= 50)
        
        # Percentile errors
        metrics['p90_error'] = np.percentile(errors, 90)
        metrics['p95_error'] = np.percentile(errors, 95)
        
        return metrics
    
    def select_best_model(self, metric: str = 'rmse', minimize: bool = True):
        """Select best model based on validation performance"""
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            raise ValueError("No models trained successfully")
        
        # Find best model
        if minimize:
            best_model_type = min(valid_results.items(), key=lambda x: x[1][metric])[0]
        else:
            best_model_type = max(valid_results.items(), key=lambda x: x[1][metric])[0]
        
        self.best_model = self.models[best_model_type]
        
        logger.info(f"\nBest model: {best_model_type}")
        logger.info(f"Best {metric}: {valid_results[best_model_type][metric]:.4f}")
        
        return best_model_type
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive evaluation on test set"""
        if self.best_model is None:
            raise ValueError("No best model selected")
        
        logger.info("\nEvaluating best model on test set...")
        
        # Basic metrics
        test_metrics = self._evaluate_model(self.best_model, X_test, y_test)
        
        # Get predictions for detailed analysis
        y_pred = self.best_model.predict(X_test)
        
        # Residual analysis
        residuals = y_test - y_pred
        test_metrics['residual_analysis'] = {
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'skew_residual': float(pd.Series(residuals).skew()),
            'kurtosis_residual': float(pd.Series(residuals).kurtosis())
        }
        
        # Prediction bounds if available
        if hasattr(self.best_model, 'predict_with_bounds'):
            bounds = self.best_model.predict_with_bounds(X_test)
            coverage = np.mean(
                (y_test >= bounds['lower_bound']) & 
                (y_test <= bounds['upper_bound'])
            )
            test_metrics['prediction_interval_coverage'] = coverage
        
        # Feature importance
        importance_df = self.best_model.get_feature_importance()
        if importance_df is not None:
            test_metrics['top_features'] = importance_df.head(10).to_dict()
        
        # Store predictions for plotting
        self.test_predictions = {
            'y_true': y_test,
            'y_pred': y_pred,
            'residuals': residuals
        }
        
        return test_metrics
    
    def plot_results(self, save_path: Optional[Path] = None):
        """Generate diagnostic plots"""
        if not hasattr(self, 'test_predictions'):
            logger.warning("No test predictions available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        y_true = self.test_predictions['y_true']
        y_pred = self.test_predictions['y_pred']
        residuals = self.test_predictions['residuals']
        
        # 1. Predicted vs Actual
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel(f'Actual {self.target_type.title()}')
        ax.set_ylabel(f'Predicted {self.target_type.title()}')
        ax.set_title('Predicted vs Actual')
        
        # 2. Residual plot
        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel(f'Predicted {self.target_type.title()}')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        
        # 3. Residual distribution
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        
        # 4. Q-Q plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plots saved to {save_path}")
        
        return fig
    
    def save_models(self, save_all: bool = False):
        """Save trained models and artifacts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        SAVED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
        SCALERS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        scaler_path = SCALERS_PATH / f'{self.target_type}_regressor_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        if save_all:
            # Save all models
            for model_type, model in self.models.items():
                if model is not None:
                    model_path = SAVED_MODELS_PATH / f'{self.target_type}_{model_type}_{timestamp}.pkl'
                    model.save_model(model_path)
        else:
            # Save only best model
            if self.best_model is not None:
                model_path = SAVED_MODELS_PATH / f'{self.target_type}_best_regressor.pkl'
                self.best_model.save_model(model_path)
                logger.info(f"Best model saved to {model_path}")
        
        # Save results
        results_path = SAVED_MODELS_PATH / f'{self.target_type}_regression_results_{timestamp}.json'
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for model_type, metrics in self.results.items():
            if isinstance(metrics, dict):
                json_results[model_type] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in metrics.items()
                }
            else:
                json_results[model_type] = metrics
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Results saved to {results_path}")
    
    def run_full_pipeline(self, save_models: bool = True,
                         generate_plots: bool = True) -> Dict[str, Any]:
        """Run complete regression training pipeline"""
        logger.info(f"Starting {self.target_type} regression training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, X_val, y_val = self.prepare_data()
        
        # Train models
        self.train_models(X_train, y_train, X_val, y_val)
        
        # Select best model
        best_model_type = self.select_best_model()
        
        # Evaluate on test set
        test_results = self.evaluate_on_test_set(X_test, y_test)
        
        # Generate plots
        if generate_plots:
            plot_path = SAVED_MODELS_PATH / f'{self.target_type}_diagnostic_plots.png'
            self.plot_results(save_path=plot_path)
        
        # Save models
        if save_models:
            self.save_models()
        
        # Generate report
        report = {
            'target_type': self.target_type,
            'era_strategy': self.era_strategy,
            'model_types': self.model_types,
            'feature_count': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'model_results': self.results,
            'best_model': best_model_type,
            'test_results': test_results,
            'data_statistics': {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'train_years': self.train_years.tolist(),
                'test_years': self.test_years.tolist()
            }
        }
        
        logger.info("\nTraining pipeline complete!")
        logger.info(f"Best model ({best_model_type}) test RMSE: {test_results['rmse']:.2f}")
        logger.info(f"Best model test R²: {test_results['r2']:.4f}")
        
        return report


# Convenience function
def train_wins_regressor(era_strategy: str = 'modern',
                        model_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """Quick function to train wins regressor"""
    trainer = RegressorTrainer(
        target_type='wins',
        era_strategy=era_strategy,
        model_types=model_types
    )
    
    return trainer.run_full_pipeline()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Train wins predictor
    report = train_wins_regressor()
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Best Model: {report['best_model']}")
    print(f"Test RMSE: {report['test_results']['rmse']:.2f}")
    print(f"Test R²: {report['test_results']['r2']:.4f}")