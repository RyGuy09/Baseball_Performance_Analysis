"""
Helper functions for MLB Team Success Predictor
"""

import os
import json
import pickle
import random
import logging
import functools
import time
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# Random seed management
def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set seeds for common ML libraries
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seed set to {seed}")


# Logging utilities
def create_logger(name: str, 
                 level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Create a configured logger
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Optional format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Timing utilities
@contextmanager
def timer(name: str = "Operation", logger_func: Optional[Callable] = None):
    """
    Context manager for timing operations
    
    Args:
        name: Name of the operation
        logger_func: Optional logging function (defaults to print)
        
    Example:
        with timer("Model training"):
            model.fit(X, y)
    """
    if logger_func is None:
        logger_func = logger.info if logger else print
    
    start_time = time.time()
    logger_func(f"{name} started...")
    
    yield
    
    elapsed_time = time.time() - start_time
    
    if elapsed_time < 60:
        logger_func(f"{name} completed in {elapsed_time:.2f} seconds")
    else:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        logger_func(f"{name} completed in {minutes}m {seconds:.2f}s")


def timeit(func: Callable) -> Callable:
    """
    Decorator for timing function execution
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.2f} seconds")
        return result
    return wrapper


# Memory utilities
@contextmanager
def memory_usage(name: str = "Operation"):
    """
    Context manager for tracking memory usage
    
    Args:
        name: Name of the operation
    """
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = end_memory - start_memory
    
    logger.info(f"{name} used {memory_used:.2f} MB of memory")


# File system utilities
def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: File path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=convert_numpy)
    
    logger.debug(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: File path
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]):
    """
    Save data to pickle file
    
    Args:
        data: Data to save
        filepath: File path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    logger.debug(f"Saved pickle to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file
    
    Args:
        filepath: File path
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# Data validation utilities
def validate_dataframe(df: pd.DataFrame, 
                      required_columns: List[str],
                      min_rows: int = 1) -> bool:
    """
    Validate DataFrame structure
    
    Args:
        df: DataFrame to validate
        required_columns: Required column names
        min_rows: Minimum number of rows
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input is not a DataFrame")
        return False
    
    if len(df) < min_rows:
        logger.error(f"DataFrame has {len(df)} rows, minimum is {min_rows}")
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        return False
    
    return True


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data quality and return summary
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_values': {},
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    # Check missing values
    missing = df.isnull().sum()
    quality_report['missing_values'] = missing[missing > 0].to_dict()
    quality_report['missing_percentage'] = {
        col: (count / len(df)) * 100 
        for col, count in quality_report['missing_values'].items()
    }
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        quality_report['numeric_summary'] = {
            col: {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'zeros': (df[col] == 0).sum(),
                'negative': (df[col] < 0).sum()
            }
            for col in numeric_cols
        }
    
    return quality_report


# Metric calculation utilities
def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     task_type: str = 'classification',
                     y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive metrics
    
    Args:
        y_true: True labels
        y_pred: Predictions
        task_type: 'classification' or 'regression'
        y_proba: Predicted probabilities (classification only)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if task_type == 'classification':
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix
        )
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle binary and multiclass
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics if probabilities available
        if y_proba is not None and len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
            
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_proba)
            metrics['log_loss'] = log_loss(y_true, y_proba)
    
    else:  # regression
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            mean_absolute_percentage_error, median_absolute_error
        )
        
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        metrics['median_ae'] = median_absolute_error(y_true, y_pred)
        
        # Additional regression metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_error'] = np.max(np.abs(residuals))
    
    return metrics


# Data transformation utilities
def normalize_team_names(df: pd.DataFrame, 
                        team_column: str = 'team_name') -> pd.DataFrame:
    """
    Normalize team names to current names
    
    Args:
        df: DataFrame with team names
        team_column: Column containing team names
        
    Returns:
        DataFrame with normalized team names
    """
    from .constants import get_current_team_name
    
    df = df.copy()
    df[team_column] = df[team_column].apply(get_current_team_name)
    
    return df


def create_train_test_split_by_year(df: pd.DataFrame,
                                   test_years: int = 3,
                                   year_column: str = 'year') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by year for time series evaluation
    
    Args:
        df: DataFrame with year column
        test_years: Number of recent years for test set
        year_column: Column containing years
        
    Returns:
        Tuple of (train_df, test_df)
    """
    max_year = df[year_column].max()
    test_start_year = max_year - test_years + 1
    
    train_df = df[df[year_column] < test_start_year].copy()
    test_df = df[df[year_column] >= test_start_year].copy()
    
    logger.info(f"Train years: {train_df[year_column].min()}-{train_df[year_column].max()}")
    logger.info(f"Test years: {test_df[year_column].min()}-{test_df[year_column].max()}")
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    return train_df, test_df


# Model utilities
def get_model_size(model: Any) -> float:
    """
    Get model size in MB
    
    Args:
        model: Model object
        
    Returns:
        Size in MB
    """
    import pickle
    return len(pickle.dumps(model)) / 1024 / 1024


def compare_models(models: Dict[str, Any], 
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  task_type: str = 'classification') -> pd.DataFrame:
    """
    Compare multiple models
    
    Args:
        models: Dictionary of model_name -> model
        X_test: Test features
        y_test: Test targets
        task_type: 'classification' or 'regression'
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models.items():
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if classification
        y_proba = None
        if task_type == 'classification' and hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, task_type, y_proba)
        metrics['model'] = name
        metrics['model_size_mb'] = get_model_size(model)
        
        # Timing
        start_time = time.time()
        _ = model.predict(X_test[:100])  # Predict on subset
        metrics['inference_time_ms'] = (time.time() - start_time) * 10  # Per 100 samples
        
        results.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model'] + [col for col in comparison_df.columns if col != 'model']
    comparison_df = comparison_df[cols]
    
    return comparison_df


# Visualization utilities
def create_results_summary(results: Dict[str, Any],
                          save_path: Optional[Path] = None) -> str:
    """
    Create formatted results summary
    
    Args:
        results: Dictionary of results
        save_path: Optional path to save summary
        
    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=" * 60)
    summary.append("MLB PREDICTION MODEL RESULTS SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Format results recursively
    def format_dict(d: Dict, indent: int = 0):
        lines = []
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(format_dict(value, indent + 1))
            elif isinstance(value, (list, tuple)) and len(value) > 5:
                lines.append(f"{prefix}{key}: [{len(value)} items]")
            elif isinstance(value, float):
                lines.append(f"{prefix}{key}: {value:.4f}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return lines
    
    summary.extend(format_dict(results))
    summary.append("=" * 60)
    
    summary_text = "\n".join(summary)
    
    if save_path:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        with open(save_path, 'w') as f:
            f.write(summary_text)
        logger.info(f"Summary saved to {save_path}")
    
    return summary_text


# Cache utilities
@functools.lru_cache(maxsize=128)
def cached_data_load(filepath: str) -> pd.DataFrame:
    """
    Cached data loading
    
    Args:
        filepath: Path to data file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)


# Parallel processing utilities
def parallel_apply(func: Callable, 
                  items: List[Any],
                  n_jobs: int = -1) -> List[Any]:
    """
    Apply function in parallel
    
    Args:
        func: Function to apply
        items: List of items
        n_jobs: Number of parallel jobs
        
    Returns:
        List of results
    """
    from joblib import Parallel, delayed
    
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(item) for item in items
    )
    
    return results


# Error handling utilities
def safe_divide(a: Union[float, np.ndarray], 
                b: Union[float, np.ndarray],
                fill_value: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safe division with zero handling
    
    Args:
        a: Numerator
        b: Denominator
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result of division
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        if isinstance(result, np.ndarray):
            result[~np.isfinite(result)] = fill_value
        else:
            if not np.isfinite(result):
                result = fill_value
    return result


# Date utilities
def get_season_dates(year: int) -> Tuple[datetime, datetime]:
    """
    Get approximate season start and end dates
    
    Args:
        year: Season year
        
    Returns:
        Tuple of (start_date, end_date)
    """
    # MLB season typically runs April-October
    start_date = datetime(year, 4, 1)
    end_date = datetime(year, 10, 31)
    
    return start_date, end_date


if __name__ == "__main__":
    # Test utilities
    set_random_seed(42)
    
    # Test timer
    with timer("Test operation"):
        time.sleep(0.1)
    
    # Test data quality
    test_df = pd.DataFrame({
        'team': ['Yankees', 'Red Sox', 'Yankees'],
        'wins': [95, 88, 95],
        'losses': [67, 74, None]
    })
    
    quality = check_data_quality(test_df)
    print("\nData Quality Report:")
    print(json.dumps(quality, indent=2))