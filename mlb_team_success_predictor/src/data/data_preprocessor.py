"""
Data preprocessing module

Handles data cleaning, type conversion, and preparation for feature engineering.
Includes era-based preprocessing options.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Clean and preprocess MLB statistics data"""
    
    def __init__(self, era_strategy: str = 'modern'):
        """
        Initialize preprocessor
        
        Args:
            era_strategy: 'modern' (2006+), 'historical' (all), or 'stratified'
        """
        self.era_strategy = era_strategy
        self.era_ranges = {
            'dead_ball': (1901, 1919),
            'live_ball': (1920, 1941),
            'integration': (1942, 1962),
            'expansion': (1963, 1976),
            'free_agency': (1977, 1993),
            'steroid': (1994, 2005),
            'modern': (2006, 2024)
        }
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Clean data
        df = self._clean_data(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Convert data types
        df = self._convert_data_types(df)
        
        # Parse string columns
        df = self._parse_string_columns(df)
        
        # Apply era filtering if needed
        if self.era_strategy == 'modern':
            df = self._filter_modern_era(df)
        
        # Sort by team and year for time-based features
        df = df.sort_values(['team_name', 'year']).reset_index(drop=True)
        
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data issues"""
        # Remove any completely duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Strip whitespace from string columns
        str_columns = df.select_dtypes(include=['object']).columns
        for col in str_columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Replace empty strings with NaN
        df = df.replace('', np.nan)
        
        # Handle specific known data issues
        # Example: Some team names might have variations
        team_name_mapping = {
            'California Angels': 'Los Angeles Angels',
            'Anaheim Angels': 'Los Angeles Angels',
            'Florida Marlins': 'Miami Marlins',
            'Montreal Expos': 'Washington Nationals'
        }
        df['team_name'] = df['team_name'].replace(team_name_mapping)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        # Division winners have NaN for games_behind
        # This is expected and will be used as a feature
        
        # For win-loss records that might be missing
        if 'record_in_the_last_10_games' in df.columns:
            df['record_in_the_last_10_games'] = df['record_in_the_last_10_games'].fillna('5-5')
        
        # For streaks
        if 'current_streak' in df.columns:
            df['current_streak'] = df['current_streak'].fillna('W0')
        
        # Log missing value summary
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        if len(missing_cols) > 0:
            logger.info("Missing values summary:")
            for col, count in missing_cols.items():
                pct = (count / len(df)) * 100
                logger.info(f"  - {col}: {count} ({pct:.1f}%)")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types"""
        # Integer columns
        int_columns = ['year', 'wins', 'losses', 'runs_scored', 'runs_allowed', 
                      'run_differential']
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Float columns
        float_columns = ['winning_percentage', 'games_behind', 'wild_card_games_behind']
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _parse_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse string columns to extract useful information"""
        # Parse record strings (e.g., "50-31" -> wins=50, losses=31)
        record_columns = ['record_at_home', 'record_when_away', 
                         'record_against_top_50_percent', 'expected_win_loss_record']
        
        for col in record_columns:
            if col in df.columns:
                df = self._parse_record_column(df, col)
        
        # Parse last 10 games record
        if 'record_in_the_last_10_games' in df.columns:
            df = self._parse_last_10_games(df)
        
        # Parse current streak
        if 'current_streak' in df.columns:
            df = self._parse_streak(df)
        
        return df
    
    def _parse_record_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Parse a win-loss record column"""
        def parse_record(record_str):
            if pd.isna(record_str):
                return pd.Series([np.nan, np.nan])
            
            match = re.match(r'(\d+)-(\d+)', str(record_str))
            if match:
                wins = int(match.group(1))
                losses = int(match.group(2))
                return pd.Series([wins, losses])
            else:
                return pd.Series([np.nan, np.nan])
        
        # Create new columns for wins and losses
        prefix = col.replace('record_', '').replace('_', '_')
        wins_col = f"{prefix}_wins"
        losses_col = f"{prefix}_losses"
        
        df[[wins_col, losses_col]] = df[col].apply(parse_record)
        
        return df
    
    def _parse_last_10_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse last 10 games record"""
        def parse_last_10(record_str):
            if pd.isna(record_str):
                return pd.Series([5, 5])  # Default to .500
            
            match = re.match(r'(\d+)-(\d+)', str(record_str))
            if match:
                wins = int(match.group(1))
                losses = int(match.group(2))
                return pd.Series([wins, losses])
            else:
                return pd.Series([5, 5])
        
        df[['last_10_wins', 'last_10_losses']] = df['record_in_the_last_10_games'].apply(
            parse_last_10
        )
        df['last_10_pct'] = df['last_10_wins'] / 10.0
        
        return df
    
    def _parse_streak(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse current streak information"""
        def parse_streak(streak_str):
            if pd.isna(streak_str):
                return pd.Series(['W', 0])
            
            streak_str = str(streak_str).strip()
            if not streak_str:
                return pd.Series(['W', 0])
            
            # Extract direction (W or L) and length
            direction = streak_str[0]
            try:
                length = int(streak_str[1:])
            except:
                length = 0
            
            return pd.Series([direction, length])
        
        df[['streak_type', 'streak_length']] = df['current_streak'].apply(parse_streak)
        
        # Create binary indicator for winning streak
        df['on_winning_streak'] = (df['streak_type'] == 'W').astype(int)
        
        return df
    
    def _filter_modern_era(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to modern era (2006-2024) if specified"""
        start_year, end_year = self.era_ranges['modern']
        mask = (df['year'] >= start_year) & (df['year'] <= end_year)
        
        modern_df = df[mask].copy()
        logger.info(f"Filtered to modern era ({start_year}-{end_year}): "
                   f"{len(df)} -> {len(modern_df)} rows")
        
        return modern_df
    
    def get_era_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data by era for stratified analysis"""
        era_splits = {}
        
        for era_name, (start_year, end_year) in self.era_ranges.items():
            mask = (df['year'] >= start_year) & (df['year'] <= end_year)
            era_df = df[mask]
            
            if len(era_df) > 0:
                era_splits[era_name] = era_df
                logger.info(f"{era_name} era: {len(era_df)} rows "
                           f"({start_year}-{end_year})")
        
        return era_splits
    
    def create_era_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add era classification column"""
        df = df.copy()
        
        def classify_era(year):
            for era_name, (start_year, end_year) in self.era_ranges.items():
                if start_year <= year <= end_year:
                    return era_name
            return 'unknown'
        
        df['era'] = df['year'].apply(classify_era)
        
        return df


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example preprocessing
    preprocessor = DataPreprocessor(era_strategy='modern')
    
    # Create sample data
    sample_data = pd.DataFrame({
        'team_name': ['Team A', 'Team B'],
        'year': [2020, 2020],
        'wins': ['95', '85'],
        'losses': ['67', '77'],
        'record_at_home': ['50-31', '45-36'],
        'current_streak': ['W3', 'L2']
    })
    
    print("Original data:")
    print(sample_data)
    print("\nPreprocessed data:")
    processed = preprocessor.preprocess(sample_data)
    print(processed)