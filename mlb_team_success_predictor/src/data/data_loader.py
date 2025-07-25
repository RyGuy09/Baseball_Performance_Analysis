"""
Data loading and validation module

Handles loading MLB statistics data from CSV files and performing
initial validation checks.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging
from pathlib import Path

from ..utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and perform initial data validation for MLB statistics"""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the CSV file. If None, uses default from config
        """
        self.data_path = data_path or RAW_DATA_PATH
        self.df = None
        self._required_columns = [
            'team_name', 'year', 'wins', 'losses', 'winning_percentage',
            'games_behind', 'wild_card_games_behind', 'runs_scored', 
            'runs_allowed', 'run_differential'
        ]
        
    def load_data(self) -> pd.DataFrame:
        """
        Load raw MLB statistics data from CSV
        
        Returns:
            pd.DataFrame: Raw MLB statistics data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data file is empty or corrupted
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            # Check if file exists
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Load data
            self.df = pd.read_csv(self.data_path)
            
            # Basic validation
            if self.df.empty:
                raise ValueError("Data file is empty")
                
            logger.info(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df: Optional[pd.DataFrame] = None) -> Tuple[bool, List[str]]:
        """
        Validate data integrity and structure
        
        Args:
            df: DataFrame to validate. If None, uses self.df
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if df is None:
            df = self.df
            
        if df is None:
            return False, ["No data loaded"]
            
        issues = []
        
        # Check required columns
        missing_cols = set(self._required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        numeric_cols = ['year', 'wins', 'losses', 'winning_percentage', 
                       'runs_scored', 'runs_allowed', 'run_differential']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' should be numeric")
        
        # Check for duplicate team-year combinations
        if 'team_name' in df.columns and 'year' in df.columns:
            duplicates = df.duplicated(subset=['team_name', 'year'])
            if duplicates.any():
                n_duplicates = duplicates.sum()
                issues.append(f"Found {n_duplicates} duplicate team-year combinations")
        
        # Check year range
        if 'year' in df.columns:
            min_year = df['year'].min()
            max_year = df['year'].max()
            
            if min_year < 1901:
                issues.append(f"Data contains years before 1901 (earliest: {min_year})")
            
            if max_year > 2025:
                issues.append(f"Data contains future years (latest: {max_year})")
        
        # Check wins + losses = total games (with some flexibility for incomplete seasons)
        if all(col in df.columns for col in ['wins', 'losses']):
            total_games = df['wins'] + df['losses']
            
            # Most seasons should have 154 or 162 games (excluding shortened seasons)
            unusual_seasons = df[(total_games < 100) | (total_games > 165)]
            if len(unusual_seasons) > 0:
                logger.warning(f"Found {len(unusual_seasons)} seasons with unusual game counts")
        
        # Check for missing values in critical columns
        critical_cols = ['team_name', 'year', 'wins', 'losses']
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"Column '{col}' has {null_count} missing values")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
                
        return is_valid, issues
    
    def load_and_validate(self) -> pd.DataFrame:
        """
        Load data and perform validation
        
        Returns:
            pd.DataFrame: Validated data
            
        Raises:
            ValueError: If validation fails
        """
        df = self.load_data()
        is_valid, issues = self.validate_data(df)
        
        if not is_valid:
            raise ValueError(f"Data validation failed: {'; '.join(issues)}")
            
        return df
    
    def get_data_info(self) -> Dict[str, any]:
        """
        Get summary information about the loaded data
        
        Returns:
            Dictionary containing data summary statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        info = {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'year_range': (self.df['year'].min(), self.df['year'].max()),
            'n_teams': self.df['team_name'].nunique(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        return info
    
    def filter_by_year_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Filter data by year range
        
        Args:
            start_year: First year to include
            end_year: Last year to include
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        mask = (self.df['year'] >= start_year) & (self.df['year'] <= end_year)
        filtered_df = self.df[mask].copy()
        
        logger.info(f"Filtered data from {start_year} to {end_year}: "
                   f"{len(filtered_df)} rows remaining")
        
        return filtered_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'processed_data.csv'):
        """
        Save processed data to the processed data directory
        
        Args:
            df: DataFrame to save
            filename: Name of the output file
        """
        output_path = PROCESSED_DATA_PATH / filename
        
        # Create directory if it doesn't exist
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = DataLoader()
    
    # Load and validate data
    try:
        df = loader.load_and_validate()
        print("\nData loaded successfully!")
        
        # Get data info
        info = loader.get_data_info()
        print("\nData Summary:")
        print(f"  - Shape: {info['n_rows']} rows × {info['n_columns']} columns")
        print(f"  - Years: {info['year_range'][0]} to {info['year_range'][1]}")
        print(f"  - Teams: {info['n_teams']}")
        print(f"  - Memory: {info['memory_usage_mb']:.2f} MB")
        
        # Filter to modern era
        modern_df = loader.filter_by_year_range(2006, 2024)
        print(f"\nModern era data: {len(modern_df)} rows")
        
    except Exception as e:
        print(f"Error: {e}")