"""
Hyperparameter optimization for MLB prediction models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import optuna
from typing import Dict, List, Any, Optional, Callable, Union
import logging
import json
from datetime import datetime
from pathlib import Path

from ..data.models.classification_models import DivisionWinnerClassifier
from ..data.models.regression_models import WinsRegressor
from ..data.utils.config import SAVED_MODELS_PATH

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter optimization for MLB models"""
    
    def __init__(self, model_class: str, model_type: str,
                 search_method: str = 'bayesian',
                 n_iter: int = 50,
                 cv_folds: int = 5,
                 scoring: Optional[str] = None,
                 n_jobs: int = -1):
        """
        Initialize hyperparameter tuner
        
        Args:
            model_class: 'classification' or 'regression'
            model_type: Specific model type (e.g., 'random_forest', 'xgboost')
            search_method: 'grid', 'random', 'bayesian', or 'optuna'
            n_iter: Number of iterations for random/bayesian search
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric (default based on model_class)
            n_jobs: Number of parallel jobs
        """
        self.model_class = model_class
        self.model_type = model_type
        self.search_method = search_method
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        
        # Set default scoring
        if scoring is None:
            self.scoring = 'roc_auc' if model_class == 'classification' else 'neg_mean_squared_error'
        else:
            self.scoring = scoring
        
        # Storage
        self.search_spaces = {}
        self.best_params = None
        self.best_score = None
        self.search_results = None
        
    def get_search_space(self) -> Dict[str, Any]:
        """Get hyperparameter search space for each model type"""
        if self.search_method == 'grid':
            return self._get_grid_search_space()
        elif self.search_method in ['random', 'bayesian']:
            return self._get_random_search_space()
        elif self.search_method == 'optuna':
            return self._get_optuna_search_space()
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")
    
    def _get_grid_search_space(self) -> Dict[str, List[Any]]:
        """Grid search parameter spaces"""
        spaces = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.3]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.1, 0.3],
                'feature_fraction': [0.6, 0.8, 1.0],
                'bagging_fraction': [0.6, 0.8, 1.0],
                'min_child_samples': [5, 10, 20]
            },
            'elastic_net': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        return spaces.get(self.model_type, {})
    
    def _get_random_search_space(self) -> Dict[str, Any]:
        """Random/Bayesian search parameter distributions"""
        if self.search_method == 'bayesian':
            # Bayesian optimization spaces
            spaces = {
                'random_forest': {
                    'n_estimators': Integer(50, 300),
                    'max_depth': Integer(3, 30),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'max_features': Categorical(['sqrt', 'log2', None])
                },
                'xgboost': {
                    'n_estimators': Integer(50, 300),
                    'max_depth': Integer(3, 15),
                    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
                    'subsample': Real(0.5, 1.0),
                    'colsample_bytree': Real(0.5, 1.0),
                    'gamma': Real(0, 0.5),
                    'min_child_weight': Integer(1, 10)
                },
                'lightgbm': {
                    'n_estimators': Integer(50, 300),
                    'num_leaves': Integer(10, 100),
                    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
                    'feature_fraction': Real(0.5, 1.0),
                    'bagging_fraction': Real(0.5, 1.0),
                    'min_child_samples': Integer(5, 50),
                    'lambda_l1': Real(0, 10),
                    'lambda_l2': Real(0, 10)
                }
            }
        else:
            # Random search distributions
            from scipy.stats import randint, uniform
            spaces = {
                'random_forest': {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 30),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None]
                },
                'xgboost': {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 15),
                    'learning_rate': uniform(0.01, 0.29),
                    'subsample': uniform(0.5, 0.5),
                    'colsample_bytree': uniform(0.5, 0.5),
                    'gamma': uniform(0, 0.5)
                }
            }
        
        return spaces.get(self.model_type, {})
    
    def _get_optuna_search_space(self) -> Callable:
        """Optuna search space as objective function"""
        def objective(trial):
            if self.model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
            elif self.model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                }
            elif self.model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
                }
            else:
                params = {}
            
            return params
        
        return objective
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting {self.search_method} hyperparameter search for {self.model_type}...")
        
        # Get base model
        if self.model_class == 'classification':
            base_model = DivisionWinnerClassifier(model_type=self.model_type).model
        else:
            base_model = WinsRegressor(model_type=self.model_type).model
        
        # Run optimization based on method
        if self.search_method == 'optuna':
            results = self._run_optuna_search(X_train, y_train, X_val, y_val)
        else:
            results = self._run_sklearn_search(base_model, X_train, y_train)
        
        return results
    
    def _run_sklearn_search(self, base_model: Any, X_train: np.ndarray, 
                           y_train: np.ndarray) -> Dict[str, Any]:
        """Run sklearn-based hyperparameter search"""
        search_space = self.get_search_space()
        
        if self.search_method == 'grid':
            searcher = GridSearchCV(
                base_model,
                search_space,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=2
            )
        elif self.search_method == 'random':
            searcher = RandomizedSearchCV(
                base_model,
                search_space,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=2,
                random_state=42
            )
        elif self.search_method == 'bayesian':
            searcher = BayesSearchCV(
                base_model,
                search_space,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=2,
                random_state=42
            )
        
        # Fit searcher
        searcher.fit(X_train, y_train)
        
        # Store results
        self.best_params = searcher.best_params_
        self.best_score = searcher.best_score_
        self.search_results = pd.DataFrame(searcher.cv_results_)
        
        # Prepare results dictionary
        results = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'search_method': self.search_method,
            'model_type': self.model_type,
            'n_iterations': len(self.search_results),
            'search_time': datetime.now().isoformat()
        }
        
        # Add top 5 parameter combinations
        top_5 = self.search_results.nlargest(5, 'mean_test_score')[
            ['params', 'mean_test_score', 'std_test_score']
        ].to_dict('records')
        results['top_5_params'] = top_5
        
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return results
    
    def _run_optuna_search(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run Optuna hyperparameter optimization"""
        import optuna.integration.lightgbm as lgb_optuna
        
        def objective(trial):
            # Get parameters from trial
            params = self._get_optuna_search_space()(trial)
            
            # Create model
            if self.model_class == 'classification':
                model = DivisionWinnerClassifier(
                    model_type=self.model_type,
                    model_params=params
                ).model
            else:
                model = WinsRegressor(
                    model_type=self.model_type,
                    model_params=params
                ).model
            
            # Cross-validation or validation set evaluation
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train)
                if self.model_class == 'classification':
                    from sklearn.metrics import roc_auc_score
                    y_pred = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred)
                else:
                    from sklearn.metrics import mean_squared_error
                    y_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, y_pred)
            else:
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=1
                )
                score = scores.mean()
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction='maximize' if 'neg_' not in self.scoring else 'minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_iter, show_progress_bar=True)
        
        # Store results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Prepare results
        results = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'search_method': 'optuna',
            'model_type': self.model_type,
            'n_trials': len(study.trials),
            'search_time': datetime.now().isoformat()
        }
        
        # Add trial history
        trial_data = []
        for trial in study.trials:
            trial_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            })
        results['trial_history'] = trial_data[:10]  # First 10 trials
        
        # Feature importance from Optuna
        try:
            importance = optuna.importance.get_param_importances(study)
            results['param_importance'] = importance
        except:
            pass
        
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return results
    
    def save_results(self, save_path: Optional[Path] = None):
        """Save optimization results"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = SAVED_MODELS_PATH / f'hyperparam_results_{self.model_type}_{timestamp}.json'
        
        results = {
            'model_class': self.model_class,
            'model_type': self.model_type,
            'search_method': self.search_method,
            'scoring': self.scoring,
            'best_params': self.best_params,
            'best_score': float(self.best_score) if self.best_score is not None else None,
            'n_iterations': self.n_iter,
            'cv_folds': self.cv_folds
        }
        
        # Add search results if available
        if self.search_results is not None:
            results['search_summary'] = self.search_results.head(10).to_dict()
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")


# Convenience function
def tune_model(model_class: str, model_type: str,
               X_train: np.ndarray, y_train: np.ndarray,
               search_method: str = 'bayesian',
               n_iter: int = 50) -> Dict[str, Any]:
    """
    Quick function to tune a model
    
    Args:
        model_class: 'classification' or 'regression'
        model_type: Model type (e.g., 'xgboost')
        X_train: Training features
        y_train: Training targets
        search_method: Search method
        n_iter: Number of iterations
        
    Returns:
        Dictionary with best parameters and results
    """
    tuner = HyperparameterTuner(
        model_class=model_class,
        model_type=model_type,
        search_method=search_method,
        n_iter=n_iter
    )
    
    results = tuner.optimize(X_train, y_train)
    tuner.save_results()
    
    return results