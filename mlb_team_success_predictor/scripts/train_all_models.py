#!/usr/bin/env python3
"""
Train all models for MLB Team Success Predictor

This script trains all required models (classification, regression, and milestone)
and saves them to the models directory.

Usage:
    python scripts/train_all_models.py [options]

Options:
    --era {all,modern,recent}  Era strategy for training data
    --cv-folds N              Number of cross-validation folds
    --test-year YEAR          Year to use for testing
    --tune-hyperparams        Run hyperparameter tuning
    --verbose                 Enable verbose output
"""

import argparse
import sys
import logging
from pathlib import Path
import json
from datetime import datetime
import time
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.training.train_classifier import ClassifierTrainer
from src.training.train_regressor import RegressorTrainer
from src.models.milestone_predictor import MilestonePredictor
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.utils.helpers import set_random_seed, ensure_dir
from src.utils.config import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS, 
    CLASSIFICATION_FEATURES, REGRESSION_FEATURES
)


def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train all MLB prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--era',
        type=str,
        choices=['all', 'modern', 'recent'],
        default='modern',
        help='Era strategy for training data (default: modern)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    parser.add_argument(
        '--test-year',
        type=int,
        default=2024,
        help='Year to use for testing (default: 2024)'
    )
    
    parser.add_argument(
        '--tune-hyperparams',
        action='store_true',
        help='Run hyperparameter tuning (takes longer)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['classification', 'regression', 'milestone', 'all'],
        default=['all'],
        help='Which models to train (default: all)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def load_and_prepare_data(era_strategy: str, test_year: int, logger):
    """Load and prepare data for training"""
    logger.info("Loading data...")
    
    # Load data
    loader = DataLoader()
    df = loader.load_and_validate()
    
    if df is None:
        raise ValueError("Failed to load data")
    
    logger.info(f"Loaded {len(df)} records from {df['year'].min()} to {df['year'].max()}")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor(era_strategy=era_strategy)
    df_processed = preprocessor.preprocess(df)
    
    # Engineer features
    logger.info("Engineering features...")
    engineer = FeatureEngineer(include_era_features=True)
    df_final = engineer.engineer_features(df_processed)
    
    # Split by year for time-based validation
    train_data = df_final[df_final['year'] < test_year - 1]
    val_data = df_final[df_final['year'] == test_year - 1]
    test_data = df_final[df_final['year'] >= test_year]
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data, engineer.feature_names


def train_classification_models(train_data, val_data, test_data, 
                              tune_hyperparams, cv_folds, logger):
    """Train division winner classification models"""
    logger.info("=" * 50)
    logger.info("Training Classification Models")
    logger.info("=" * 50)
    
    # Initialize trainer
    trainer = ClassifierTrainer(
        task_type='division_winner',
        model_types=CLASSIFICATION_MODELS
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_val, y_val = trainer.prepare_data(
        train_df=train_data,
        val_df=val_data,
        test_df=test_data,
        custom_features=CLASSIFICATION_FEATURES
    )
    
    logger.info(f"Features used: {len(trainer.feature_names)}")
    logger.info(f"Class distribution - Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Train models
    start_time = time.time()
    trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Hyperparameter tuning if requested
    if tune_hyperparams:
        logger.info("Running hyperparameter tuning...")
        for model_name in CLASSIFICATION_MODELS:
            if model_name in ['xgboost', 'lightgbm']:  # Tune only tree-based models
                tuner = HyperparameterTuner(
                    model_class='classification',
                    model_type=model_name,
                    search_method='random',
                    n_iter=20,
                    cv_folds=cv_folds
                )
                
                # Combine train and validation for tuning
                X_tune = np.vstack([X_train, X_val])
                y_tune = np.hstack([y_train, y_val])
                
                tuning_results = tuner.optimize(X_tune, y_tune)
                logger.info(f"{model_name} best params: {tuning_results['best_params']}")
                
                # Retrain with best params
                trainer.models[model_name].set_params(**tuning_results['best_params'])
                trainer.models[model_name].train(X_train, y_train, X_val, y_val)
    
    # Evaluate and select best model
    best_model = trainer.select_best_model(metric='roc_auc')
    test_results = trainer.evaluate_on_test_set(X_test, y_test)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.1f} seconds")
    logger.info(f"Best model: {best_model}")
    logger.info(f"Test performance: {test_results}")
    
    # Save models
    trainer.save_models(save_all=False)  # Save only best model
    
    return trainer, test_results


def train_regression_models(train_data, val_data, test_data, 
                          tune_hyperparams, cv_folds, logger):
    """Train win total regression models"""
    logger.info("=" * 50)
    logger.info("Training Regression Models")
    logger.info("=" * 50)
    
    # Initialize trainer
    trainer = RegressorTrainer(
        target_type='wins',
        model_types=REGRESSION_MODELS
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_val, y_val = trainer.prepare_data(
        train_df=train_data,
        val_df=val_data,
        test_df=test_data,
        custom_features=REGRESSION_FEATURES
    )
    
    logger.info(f"Features used: {len(trainer.feature_names)}")
    logger.info(f"Target stats - Train: mean={y_train.mean():.1f}, std={y_train.std():.1f}")
    
    # Train models
    start_time = time.time()
    trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Hyperparameter tuning if requested
    if tune_hyperparams:
        logger.info("Running hyperparameter tuning...")
        for model_name in ['xgboost', 'lightgbm']:
            if model_name in REGRESSION_MODELS:
                tuner = HyperparameterTuner(
                    model_class='regression',
                    model_type=model_name,
                    search_method='random',
                    n_iter=20,
                    cv_folds=cv_folds
                )
                
                X_tune = np.vstack([X_train, X_val])
                y_tune = np.hstack([y_train, y_val])
                
                tuning_results = tuner.optimize(X_tune, y_tune)
                logger.info(f"{model_name} best params: {tuning_results['best_params']}")
                
                # Retrain with best params
                trainer.models[model_name].set_params(**tuning_results['best_params'])
                trainer.models[model_name].train(X_train, y_train, X_val, y_val)
    
    # Evaluate and select best model
    best_model = trainer.select_best_model(metric='rmse', minimize=True)
    test_results = trainer.evaluate_on_test_set(X_test, y_test)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.1f} seconds")
    logger.info(f"Best model: {best_model}")
    logger.info(f"Test performance: {test_results}")
    
    # Save models
    trainer.save_models(save_all=False)
    
    return trainer, test_results


def train_milestone_models(train_data, val_data, test_data, logger):
    """Train milestone prediction models"""
    logger.info("=" * 50)
    logger.info("Training Milestone Models")
    logger.info("=" * 50)
    
    # Initialize trainer
    trainer = MilestonePredictor()
    
    # Define milestones
    milestone_columns = [
        'achieved_90_wins',
        'achieved_100_wins',
        'made_playoffs',
        'scored_800_runs'
    ]
    
    # Filter to available milestones
    available_milestones = [col for col in milestone_columns if col in train_data.columns]
    logger.info(f"Training for milestones: {available_milestones}")
    
    if not available_milestones:
        logger.warning("No milestone columns found in data")
        return None, None
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_val, y_val = trainer.prepare_data(
        train_df=train_data,
        val_df=val_data,
        test_df=test_data,
        milestone_columns=available_milestones
    )
    
    logger.info(f"Features used: {len(trainer.feature_names)}")
    logger.info(f"Milestone achievement rates:")
    for i, milestone in enumerate(available_milestones):
        rate = y_train[:, i].mean()
        logger.info(f"  {milestone}: {rate:.1%}")
    
    # Train model
    start_time = time.time()
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    test_results = trainer.evaluate(X_test, y_test)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.1f} seconds")
    logger.info(f"Test performance: {test_results}")
    
    # Save model
    trainer.save_model()
    
    return trainer, test_results


def generate_training_report(results, output_path):
    """Generate comprehensive training report"""
    report = {
        'training_date': datetime.now().isoformat(),
        'configuration': {
            'era_strategy': results['config']['era_strategy'],
            'cv_folds': results['config']['cv_folds'],
            'test_year': results['config']['test_year'],
            'hyperparameter_tuning': results['config']['tune_hyperparams']
        },
        'results': {
            'classification': results.get('classification', {}),
            'regression': results.get('regression', {}),
            'milestone': results.get('milestone', {})
        },
        'training_times': results.get('times', {}),
        'feature_counts': results.get('feature_counts', {}),
        'data_summary': results.get('data_summary', {})
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    """Main training pipeline"""
    args = parse_arguments()
    
    # Set up environment
    set_random_seed(args.seed)
    ensure_dir('logs')
    ensure_dir('models/saved_models')
    ensure_dir('models/scalers')
    
    logger = setup_logging(args.verbose)
    logger.info("Starting model training pipeline")
    logger.info(f"Configuration: {vars(args)}")
    
    # Track results
    results = {
        'config': vars(args),
        'times': {},
        'feature_counts': {},
        'data_summary': {}
    }
    
    try:
        # Load and prepare data
        train_data, val_data, test_data, feature_names = load_and_prepare_data(
            args.era, args.test_year, logger
        )
        
        results['data_summary'] = {
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'total_features': len(feature_names)
        }
        
        # Determine which models to train
        models_to_train = ['classification', 'regression', 'milestone'] if 'all' in args.models else args.models
        
        # Train classification models
        if 'classification' in models_to_train:
            start_time = time.time()
            classifier_trainer, class_results = train_classification_models(
                train_data, val_data, test_data,
                args.tune_hyperparams, args.cv_folds, logger
            )
            results['classification'] = class_results
            results['times']['classification'] = time.time() - start_time
            results['feature_counts']['classification'] = len(classifier_trainer.feature_names)
        
        # Train regression models
        if 'regression' in models_to_train:
            start_time = time.time()
            regressor_trainer, reg_results = train_regression_models(
                train_data, val_data, test_data,
                args.tune_hyperparams, args.cv_folds, logger
            )
            results['regression'] = reg_results
            results['times']['regression'] = time.time() - start_time
            results['feature_counts']['regression'] = len(regressor_trainer.feature_names)
        
        # Train milestone models
        if 'milestone' in models_to_train:
            start_time = time.time()
            milestone_trainer, milestone_results = train_milestone_models(
                train_data, val_data, test_data, logger
            )
            if milestone_results:
                results['milestone'] = milestone_results
                results['times']['milestone'] = time.time() - start_time
                results['feature_counts']['milestone'] = len(milestone_trainer.feature_names)
        
        # Generate training report
        report_path = Path('models/training_report.json')
        report = generate_training_report(results, report_path)
        logger.info(f"Training report saved to {report_path}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Total training time: {sum(results['times'].values()):.1f} seconds")
        
        for model_type, metrics in results.items():
            if model_type in ['classification', 'regression', 'milestone'] and metrics:
                logger.info(f"\n{model_type.upper()} MODEL:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\nAll models trained and saved successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()