"""
Cross-validation strategies for MLB prediction models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    cross_val_score, cross_validate, cross_val_predict
)
from sklearn.metrics import make_scorer
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class CrossValidator:
    """Advanced cross-validation for MLB models"""
    
    def __init__(self, cv_strategy: str = 'stratified',
                 n_folds: int = 5,
                 shuffle: bool = True,
                 random_state: int = 42):
        """
        Initialize cross-validator
        
        Args:
            cv_strategy: 'kfold', 'stratified', 'timeseries', or 'custom'
            n_folds: Number of folds
            shuffle: Whether to shuffle data (not for timeseries)
            random_state: Random seed
        """
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.shuffle = shuffle and cv_strategy != 'timeseries'
        self.random_state = random_state
        
        # Initialize CV splitter
        self.cv_splitter = self._initialize_splitter()
        
        # Storage for results
        self.cv_results = {}
        self.fold_predictions = []
        
    def _initialize_splitter(self):
        """Initialize the appropriate CV splitter"""
        if self.cv_strategy == 'kfold':
            return KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_strategy == 'stratified':
            return StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_folds)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                      scoring: Union[str, List[str], Dict[str, Any]] = None,
                      return_train_score: bool = True,
                      return_estimator: bool = False) -> Dict[str, Any]:
        """
        Perform cross-validation on a model
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            scoring: Scoring metric(s)
            return_train_score: Whether to return training scores
            return_estimator: Whether to return fitted estimators
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Running {self.cv_strategy} cross-validation with {self.n_folds} folds...")
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=self.cv_splitter,
            scoring=scoring,
            return_train_score=return_train_score,
            return_estimator=return_estimator,
            n_jobs=-1
        )
        
        # Store raw results
        self.cv_results = cv_results
        
        # Process results
        processed_results = self._process_cv_results(cv_results, scoring)
        
        return processed_results
    
    def _process_cv_results(self, cv_results: Dict[str, np.ndarray],
                           scoring: Union[str, List[str], Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw CV results into summary statistics"""
        processed = {
            'cv_strategy': self.cv_strategy,
            'n_folds': self.n_folds,
            'timestamp': datetime.now().isoformat()
        }
        
        # Handle different scoring formats
        if isinstance(scoring, str):
            score_names = [scoring]
        elif isinstance(scoring, list):
            score_names = scoring
        elif isinstance(scoring, dict):
            score_names = list(scoring.keys())
        else:
            score_names = ['score']
        
        # Process each metric
        for metric in score_names:
            test_key = f'test_{metric}' if metric != 'score' else 'test_score'
            train_key = f'train_{metric}' if metric != 'score' else 'train_score'
            
            if test_key in cv_results:
                test_scores = cv_results[test_key]
                processed[f'{metric}_mean'] = float(np.mean(test_scores))
                processed[f'{metric}_std'] = float(np.std(test_scores))
                processed[f'{metric}_min'] = float(np.min(test_scores))
                processed[f'{metric}_max'] = float(np.max(test_scores))
                processed[f'{metric}_scores'] = test_scores.tolist()
                
                # Training scores if available
                if train_key in cv_results:
                    train_scores = cv_results[train_key]
                    processed[f'{metric}_train_mean'] = float(np.mean(train_scores))
                    processed[f'{metric}_overfit'] = float(
                        np.mean(train_scores) - np.mean(test_scores)
                    )
        
        # Fit times
        if 'fit_time' in cv_results:
            processed['fit_time_mean'] = float(np.mean(cv_results['fit_time']))
            processed['fit_time_total'] = float(np.sum(cv_results['fit_time']))
        
        return processed
    
    def validate_with_predictions(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 method: str = 'predict_proba') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cross-validated predictions
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            method: 'predict' or 'predict_proba'
            
        Returns:
            Tuple of (predictions, true_values) aligned by index
        """
        logger.info("Generating cross-validated predictions...")
        
        # Initialize arrays
        predictions = np.zeros_like(y, dtype=float)
        
        # Store fold information
        fold_info = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
            # Split data
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            
            # Train model
            model_clone = self._clone_model(model)
            model_clone.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            if method == 'predict_proba' and hasattr(model_clone, 'predict_proba'):
                fold_pred = model_clone.predict_proba(X_val_fold)[:, 1]
            else:
                fold_pred = model_clone.predict(X_val_fold)
            
            # Store predictions
            predictions[val_idx] = fold_pred
            
            # Store fold information
            fold_info.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'val_indices': val_idx
            })
        
        self.fold_predictions = fold_info
        
        return predictions, y
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model"""
        from sklearn.base import clone
        return clone(model)
    
    def plot_cv_results(self, metric: str = 'score',
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot cross-validation results"""
        if metric not in self.cv_results and f'test_{metric}' not in self.cv_results:
            raise ValueError(f"Metric {metric} not found in CV results")
        
        # Get scores
        key = f'test_{metric}' if f'test_{metric}' in self.cv_results else metric
        scores = self.cv_results[key]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Fold scores
        fold_numbers = np.arange(1, len(scores) + 1)
        ax1.bar(fold_numbers, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.axhline(y=np.mean(scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(scores):.4f}')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel(f'{metric.title()} Score')
        ax1.set_title(f'Cross-Validation Scores by Fold')
        ax1.legend()
        ax1.set_xticks(fold_numbers)
        
        # Plot 2: Score distribution
        ax2.hist(scores, bins=min(10, len(scores)), color='lightgreen', 
                 edgecolor='darkgreen', alpha=0.7)
        ax2.axvline(x=np.mean(scores), color='red', linestyle='--')
        ax2.set_xlabel(f'{metric.title()} Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Distribution of CV Scores')
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(scores):.4f}\nStd: {np.std(scores):.4f}\n'
        stats_text += f'Min: {np.min(scores):.4f}\nMax: {np.max(scores):.4f}'
        ax2.text(0.65, 0.95, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', 
                 facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"CV plot saved to {save_path}")
        
        return fig
    
    def plot_learning_curves(self, model: Any, X: np.ndarray, y: np.ndarray,
                           train_sizes: np.ndarray = None,
                           scoring: str = 'accuracy') -> plt.Figure:
        """Plot learning curves to diagnose bias/variance"""
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        logger.info("Generating learning curves...")
        
        # Calculate learning curves
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=self.cv_splitter,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot training scores
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue',
                label='Training score')
        plt.fill_between(train_sizes_abs, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color='blue')
        
        # Plot validation scores
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red',
                label='Cross-validation score')
        plt.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel(f'{scoring.title()} Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add diagnosis text
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        gap = final_train - final_val
        
        diagnosis = "Diagnosis: "
        if gap > 0.1:
            diagnosis += "High variance (overfitting)"
        elif final_val < 0.7:  # Adjust threshold based on metric
            diagnosis += "High bias (underfitting)"
        else:
            diagnosis += "Good fit"
        
        plt.text(0.02, 0.02, diagnosis, transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        return plt.gcf()


class NestedCrossValidator:
    """Nested cross-validation for unbiased model selection and evaluation"""
    
    def __init__(self, outer_cv: Any = None, inner_cv: Any = None):
        """
        Initialize nested CV
        
        Args:
            outer_cv: Outer CV splitter (for evaluation)
            inner_cv: Inner CV splitter (for hyperparameter tuning)
        """
        self.outer_cv = outer_cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.inner_cv = inner_cv or StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        self.results = []
        
    def evaluate(self, model_class: Any, param_grid: Dict[str, List[Any]],
                X: np.ndarray, y: np.ndarray,
                scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform nested cross-validation
        
        Args:
            model_class: Model class to instantiate
            param_grid: Parameter grid for tuning
            X: Features
            y: Targets
            scoring: Scoring metric
            
        Returns:
            Dictionary with nested CV results
        """
        from sklearn.model_selection import GridSearchCV
        
        logger.info("Starting nested cross-validation...")
        
        outer_scores = []
        best_params_list = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.outer_cv.split(X, y)):
            logger.info(f"Outer fold {fold_idx + 1}/{self.outer_cv.n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            inner_model = GridSearchCV(
                model_class(),
                param_grid,
                cv=self.inner_cv,
                scoring=scoring,
                n_jobs=-1
            )
            
            # Fit on training data
            inner_model.fit(X_train, y_train)
            
            # Evaluate on test fold
            test_score = inner_model.score(X_test, y_test)
            outer_scores.append(test_score)
            best_params_list.append(inner_model.best_params_)
            
            logger.info(f"  Test score: {test_score:.4f}")
            logger.info(f"  Best params: {inner_model.best_params_}")
        
        # Compile results
        results = {
            'outer_scores': outer_scores,
            'mean_score': float(np.mean(outer_scores)),
            'std_score': float(np.std(outer_scores)),
            'best_params_per_fold': best_params_list,
            'scoring': scoring
        }
        
        logger.info(f"\nNested CV complete. Mean score: {results['mean_score']:.4f} "
                   f"(+/- {results['std_score']:.4f})")
        
        return results


class TimeSeriesCrossValidator:
    """Specialized cross-validation for time series data"""
    
    def __init__(self, n_splits: int = 5,
                 gap: int = 0,
                 max_train_size: Optional[int] = None):
        """
        Initialize time series CV
        
        Args:
            n_splits: Number of splits
            gap: Gap between train and test sets (to avoid leakage)
            max_train_size: Maximum training set size
        """
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        
    def split(self, X: np.ndarray, y: np.ndarray = None,
          groups: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for time series
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Determine test size
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Calculate split point
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            train_end = test_start - self.gap
            
            # Apply max_train_size if specified
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def plot_splits(self, X: np.ndarray, y: np.ndarray = None) -> plt.Figure:
        """Visualize time series splits"""
        n_samples = len(X)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each split
        for i, (train_idx, test_idx) in enumerate(self.split(X, y)):
            # Training data
            ax.barh(i, len(train_idx), left=train_idx[0], height=0.6,
                   color='blue', alpha=0.7, label='Train' if i == 0 else '')
            
            # Test data
            ax.barh(i, len(test_idx), left=test_idx[0], height=0.6,
                   color='red', alpha=0.7, label='Test' if i == 0 else '')
            
            # Gap
            if self.gap > 0 and train_idx[-1] + 1 < test_idx[0]:
                ax.barh(i, self.gap, left=train_idx[-1] + 1, height=0.6,
                       color='gray', alpha=0.5, label='Gap' if i == 0 else '')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('CV Fold')
        ax.set_title('Time Series Cross-Validation Splits')
        ax.legend()
        ax.set_xlim(0, n_samples)
        ax.set_ylim(-0.5, self.n_splits - 0.5)
        
        plt.tight_layout()
        return fig


# Convenience functions
def perform_cv_classification(model: Any, X: np.ndarray, y: np.ndarray,
                            cv_strategy: str = 'stratified',
                            n_folds: int = 5) -> Dict[str, Any]:
    """Quick CV for classification models"""
    cv = CrossValidator(cv_strategy=cv_strategy, n_folds=n_folds)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = cv.validate_model(model, X, y, scoring=scoring)
    
    return results


def perform_cv_regression(model: Any, X: np.ndarray, y: np.ndarray,
                         cv_strategy: str = 'kfold',
                         n_folds: int = 5) -> Dict[str, Any]:
    """Quick CV for regression models"""
    cv = CrossValidator(cv_strategy=cv_strategy, n_folds=n_folds)
    
    scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    results = cv.validate_model(model, X, y, scoring=scoring)
    
    # Convert negative metrics
    for metric in ['neg_mean_squared_error', 'neg_mean_absolute_error']:
        if f'{metric}_mean' in results:
            results[f'{metric.replace("neg_", "")}_mean'] = -results[f'{metric}_mean']
            results[f'{metric.replace("neg_", "")}_std'] = results[f'{metric}_std']
    
    return results


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    
    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation
    cv = CrossValidator(cv_strategy='stratified', n_folds=5)
    results = cv.validate_model(model, X, y, scoring=['accuracy', 'roc_auc'])
    
    print("\nCross-Validation Results:")
    print(f"Accuracy: {results['accuracy_mean']:.4f} (+/- {results['accuracy_std']:.4f})")
    print(f"ROC AUC: {results['roc_auc_mean']:.4f} (+/- {results['roc_auc_std']:.4f})")
    
    # Plot results
    cv.plot_cv_results(metric='accuracy')
    plt.show()