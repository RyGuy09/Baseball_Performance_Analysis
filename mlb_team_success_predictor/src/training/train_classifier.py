"""
Training pipeline for classification models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from ..data.data_loader import DataLoader
from ..data.data_preprocessor import DataPreprocessor
from ..data.feature_engineering import FeatureEngineer
from ..data.models.classification_models import DivisionWinnerClassifier, PlayoffClassifier
from ..data.utils.config import (
    CLASSIFICATION_FEATURES,
    SAVED_MODELS_PATH,
    SCALERS_PATH,
    PROCESSED_DATA_PATH
)

logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """Comprehensive training pipeline for classification models"""
    
    def __init__(self, task_type: str = 'division_winner',
                 era_strategy: str = 'modern',
                 model_types: Optional[List[str]] = None):
        """
        Initialize classifier trainer
        
        Args:
            task_type: 'division_winner' or 'playoff'
            era_strategy: 'modern', 'historical', or 'all'
            model_types: List of model types to train
        """
        self.task_type = task_type
        self.era_strategy = era_strategy
        self.model_types = model_types or ['random_forest', 'xgboost', 'lightgbm']
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor(era_strategy=era_strategy)
        self.feature_engineer = FeatureEngineer()
        
        # Storage for results
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = None
        
    def prepare_data(self, custom_features: Optional[List[str]] = None) -> Tuple[np.ndarray, ...]:
        """
        Load and prepare data for training
        
        Args:
            custom_features: Optional custom feature list
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, X_val, y_val)
        """
        logger.info("Preparing data for classification...")
        
        # Load raw data
        df = self.data_loader.load_and_validate()
        
        # Preprocess
        df = self.preprocessor.preprocess(df)
        
        # Engineer features
        df = self.feature_engineer.engineer_features(df)
        
        # Select features
        features = custom_features or CLASSIFICATION_FEATURES
        target = 'is_division_winner' if self.task_type == 'division_winner' else 'made_playoffs'
        
        # Remove rows with missing values
        required_cols = features + [target]
        df_clean = df.dropna(subset=required_cols)
        
        logger.info(f"Clean data shape: {df_clean.shape}")
        
        # Prepare features and target
        X = df_clean[features].values
        y = df_clean[target].values
        
        # Split data - stratified to maintain class balance
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature names
        self.feature_names = features
        
        # Log class distribution
        logger.info(f"Training set - Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)}")
        logger.info(f"Validation set - Class 0: {sum(y_val == 0)}, Class 1: {sum(y_val == 1)}")
        logger.info(f"Test set - Class 0: {sum(y_test == 0)}, Class 1: {sum(y_test == 1)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_val_scaled, y_val
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray):
        """Train multiple classification models"""
        logger.info(f"Training {len(self.model_types)} model types...")
        
        for model_type in self.model_types:
            logger.info(f"\nTraining {model_type}...")
            
            try:
                # Initialize model
                if self.task_type == 'division_winner':
                    model = DivisionWinnerClassifier(
                        model_type=model_type,
                        calibrate_probabilities=True
                    )
                else:
                    model = PlayoffClassifier(model_type=model_type)
                
                # Set feature names
                model.feature_names = self.feature_names
                
                # Train
                start_time = datetime.now()
                model.train(X_train, y_train, X_val, y_val)
                train_time = (datetime.now() - start_time).total_seconds()
                
                # Store model and training time
                self.models[model_type] = model
                
                # Evaluate on validation set
                val_metrics = self._evaluate_model(model, X_val, y_val)
                val_metrics['train_time'] = train_time
                
                self.results[model_type] = val_metrics
                
                logger.info(f"{model_type} - Val AUC: {val_metrics['roc_auc']:.4f}, "
                           f"Accuracy: {val_metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                self.results[model_type] = {'error': str(e)}
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        return metrics
    
    def select_best_model(self, metric: str = 'roc_auc'):
        """Select best model based on validation performance"""
        # Filter out models with errors
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            raise ValueError("No models trained successfully")
        
        # Find best model
        best_model_type = max(valid_results.items(), key=lambda x: x[1][metric])[0]
        self.best_model = self.models[best_model_type]
        
        logger.info(f"\nBest model: {best_model_type}")
        logger.info(f"Best {metric}: {valid_results[best_model_type][metric]:.4f}")
        
        return best_model_type
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate best model on test set"""
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model() first.")
        
        logger.info("\nEvaluating best model on test set...")
        
        # Get detailed metrics
        test_metrics = self._evaluate_model(self.best_model, X_test, y_test)
        
        # Get predictions for additional analysis
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        test_metrics['classification_report'] = report
        
        # Feature importance if available
        importance_df = self.best_model.get_feature_importance()
        if importance_df is not None:
            test_metrics['top_features'] = importance_df.head(10).to_dict()
        
        # Prediction analysis
        test_metrics['prediction_distribution'] = {
            'mean_probability': float(y_proba.mean()),
            'std_probability': float(y_proba.std()),
            'predictions_class_0': int(sum(y_pred == 0)),
            'predictions_class_1': int(sum(y_pred == 1))
        }
        
        return test_metrics
    
    def save_models(self, save_all: bool = False):
        """Save trained models and artifacts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        SAVED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
        SCALERS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        scaler_path = SCALERS_PATH / f'{self.task_type}_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        if save_all:
            # Save all models
            for model_type, model in self.models.items():
                if model is not None:
                    model_path = SAVED_MODELS_PATH / f'{self.task_type}_{model_type}_{timestamp}.pkl'
                    model.save_model(model_path)
        else:
            # Save only best model
            if self.best_model is not None:
                model_path = SAVED_MODELS_PATH / f'{self.task_type}_best_model.pkl'
                self.best_model.save_model(model_path)
                logger.info(f"Best model saved to {model_path}")
        
        # Save training results
        results_path = SAVED_MODELS_PATH / f'{self.task_type}_training_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Training results saved to {results_path}")
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            'task_type': self.task_type,
            'era_strategy': self.era_strategy,
            'model_types': self.model_types,
            'feature_count': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'model_results': self.results,
            'best_model': self.select_best_model() if self.best_model is None else self.best_model.model_type,
            'feature_names': self.feature_names
        }
        
        # Add data statistics
        if hasattr(self, 'X_train'):
            report['data_statistics'] = {
                'train_samples': len(self.X_train),
                'val_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'feature_count': self.X_train.shape[1]
            }
        
        return report
    
    def run_full_pipeline(self, save_models: bool = True) -> Dict[str, Any]:
        """Run complete training pipeline"""
        logger.info("Starting full classification training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, X_val, y_val = self.prepare_data()
        
        # Store for report
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        
        # Train models
        self.train_models(X_train, y_train, X_val, y_val)
        
        # Select best model
        best_model_type = self.select_best_model()
        
        # Evaluate on test set
        test_results = self.evaluate_on_test_set(X_test, y_test)
        
        # Save models if requested
        if save_models:
            self.save_models()
        
        # Generate report
        report = self.generate_training_report()
        report['test_results'] = test_results
        
        logger.info("\nTraining pipeline complete!")
        logger.info(f"Best model ({best_model_type}) test AUC: {test_results['roc_auc']:.4f}")
        
        return report


# Convenience function for quick training
def train_division_winner_classifier(era_strategy: str = 'modern',
                                   model_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Quick function to train division winner classifier
    
    Args:
        era_strategy: 'modern', 'historical', or 'all'
        model_types: List of model types to train
        
    Returns:
        Training report dictionary
    """
    trainer = ClassifierTrainer(
        task_type='division_winner',
        era_strategy=era_strategy,
        model_types=model_types
    )
    
    return trainer.run_full_pipeline()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Train division winner classifier
    report = train_division_winner_classifier(era_strategy='modern')
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Best Model: {report['best_model']}")
    print(f"Test AUC: {report['test_results']['roc_auc']:.4f}")
    print(f"Test Accuracy: {report['test_results']['accuracy']:.4f}")