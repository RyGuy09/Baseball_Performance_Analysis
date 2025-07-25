"""
Feature engineering module

Creates derived features for model training including:
- Performance metrics
- Historical features
- Era-adjusted statistics
- Milestone indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create and transform features for MLB prediction models"""
    
    def __init__(self, include_era_features: bool = True):
        """
        Initialize feature engineer
        
        Args:
            include_era_features: Whether to create era-specific features
        """
        self.include_era_features = include_era_features
        self.feature_names = []
        self._fitted_scalers = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Core features
        df = self._create_performance_features(df)
        df = self._create_efficiency_features(df)
        df = self._create_pythagorean_features(df)
        
        # Historical features
        df = self._create_historical_features(df)
        df = self._create_trend_features(df)
        
        # Context features
        df = self._create_strength_features(df)
        df = self._create_consistency_features(df)
        
        # Era-based features
        if self.include_era_features:
            df = self._create_era_adjusted_features(df)
        
        # Target variables
        df = self._create_target_variables(df)
        
        # Record feature names
        self._record_feature_names(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        
        return df
    
    def _create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic performance features"""
        # Total games
        df['total_games'] = df['wins'] + df['losses']
        
        # Home/Away performance (if parsed from preprocessor)
        if 'at_home_wins' in df.columns and 'at_home_losses' in df.columns:
            df['home_games'] = df['at_home_wins'] + df['at_home_losses']
            df['home_win_pct'] = df['at_home_wins'] / df['home_games']
            df['home_win_pct'] = df['home_win_pct'].fillna(0.5)
        
        if 'when_away_wins' in df.columns and 'when_away_losses' in df.columns:
            df['away_games'] = df['when_away_wins'] + df['when_away_losses']
            df['away_win_pct'] = df['when_away_wins'] / df['away_games']
            df['away_win_pct'] = df['away_win_pct'].fillna(0.5)
            
            # Home/away differential
            df['home_away_diff'] = df['home_win_pct'] - df['away_win_pct']
        
        # Performance against strong teams
        if 'against_top_50_percent_wins' in df.columns:
            total_vs_top = (df['against_top_50_percent_wins'] + 
                           df['against_top_50_percent_losses'])
            df['win_pct_vs_top_teams'] = (df['against_top_50_percent_wins'] / 
                                          total_vs_top)
            df['win_pct_vs_top_teams'] = df['win_pct_vs_top_teams'].fillna(0.5)
        
        return df
    
    def _create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create run efficiency features"""
        # Runs per game
        df['runs_per_game'] = df['runs_scored'] / df['total_games']
        df['runs_allowed_per_game'] = df['runs_allowed'] / df['total_games']
        
        # Run efficiency
        df['run_efficiency'] = df['runs_scored'] / df['runs_allowed']
        df['run_efficiency'] = df['run_efficiency'].replace([np.inf, -np.inf], np.nan)
        df['run_efficiency'] = df['run_efficiency'].fillna(1.0)
        
        # Offensive and defensive ratings (relative to league average)
        df['offensive_rating'] = df.groupby('year')['runs_per_game'].transform(
            lambda x: x / x.mean()
        )
        df['defensive_rating'] = df.groupby('year')['runs_allowed_per_game'].transform(
            lambda x: x.mean() / x
        )
        
        # One-run game performance (approximation based on run differential)
        df['close_game_factor'] = 1 / (1 + np.abs(df['run_differential'] / df['total_games']))
        
        return df
    
    def _create_pythagorean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Pythagorean expectation features"""
        # Basic Pythagorean wins
        df['pythagorean_wins'] = (
            df['total_games'] * (df['runs_scored']**2) / 
            (df['runs_scored']**2 + df['runs_allowed']**2)
        )
        
        # Pythagorean winning percentage
        df['pythagorean_win_pct'] = df['pythagorean_wins'] / df['total_games']
        
        # Luck factor (actual vs expected)
        df['luck_factor'] = df['wins'] - df['pythagorean_wins']
        df['luck_pct'] = df['winning_percentage'] - df['pythagorean_win_pct']
        
        # Second-order wins (Pythagorean wins based on expected runs)
        # This would require more detailed stats, so we'll approximate
        df['performance_above_talent'] = df['luck_factor'] / df['total_games']
        
        return df
    
    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on team history"""
        # Sort by team and year
        df = df.sort_values(['team_name', 'year'])
        
        # Previous season features
        historical_cols = ['wins', 'losses', 'run_differential', 'winning_percentage',
                          'pythagorean_win_pct', 'runs_per_game', 'runs_allowed_per_game']
        
        for col in historical_cols:
            if col in df.columns:
                df[f'prev_{col}'] = df.groupby('team_name')[col].shift(1)
        
        # Multi-year rolling averages
        for window in [3, 5]:
            for col in ['wins', 'run_differential', 'winning_percentage']:
                if col in df.columns:
                    df[f'{col}_{window}yr_avg'] = (
                        df.groupby('team_name')[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .shift(1)
                        .reset_index(0, drop=True)
                    )
        
        # Year-over-year changes
        for col in ['wins', 'run_differential']:
            if col in df.columns and f'prev_{col}' in df.columns:
                df[f'{col}_change'] = df[col] - df[f'prev_{col}']
        
        # Trend features (linear regression over past seasons)
        df = self._calculate_trend_features(df)
        
        return df
    
    def _calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend features using rolling linear regression"""
        def calculate_trend(series, window=5):
            """Calculate trend coefficient"""
            if len(series) < 2:
                return np.nan
            
            valid_data = series.dropna()
            if len(valid_data) < 2:
                return np.nan
                
            x = np.arange(len(valid_data))
            try:
                slope, _, _, _, _ = stats.linregress(x, valid_data)
                return slope
            except:
                return np.nan
        
        # Calculate trends for key metrics
        trend_cols = ['wins', 'run_differential', 'winning_percentage']
        
        for col in trend_cols:
            if col in df.columns:
                df[f'{col}_trend'] = (
                    df.groupby('team_name')[col]
                    .rolling(window=5, min_periods=2)
                    .apply(calculate_trend)
                    .shift(1)
                    .reset_index(0, drop=True)
                )
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and trend features"""
        # Recent performance (if available)
        if 'last_10_pct' in df.columns:
            df['momentum'] = df['last_10_pct'] - df['winning_percentage']
            df['hot_team'] = (df['last_10_pct'] >= 0.7).astype(int)
            df['cold_team'] = (df['last_10_pct'] <= 0.3).astype(int)
        
        # Streak features (if available)
        if 'streak_length' in df.columns and 'on_winning_streak' in df.columns:
            df['streak_score'] = df['streak_length'] * (2 * df['on_winning_streak'] - 1)
            df['long_winning_streak'] = (
                (df['on_winning_streak'] == 1) & (df['streak_length'] >= 5)
            ).astype(int)
        
        # Monthly performance variation (approximation)
        # In real data, you'd calculate actual monthly splits
        df['performance_volatility'] = df.groupby('team_name')['winning_percentage'].transform(
            lambda x: x.rolling(window=10, min_periods=3).std()
        )
        
        return df
    
    def _create_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
       """Create strength of schedule and competition features"""
       # Division strength (average wins of teams in same year)
       year_avg_wins = df.groupby('year')['wins'].transform('mean')
       df['league_strength'] = year_avg_wins / df['total_games']
       
       # Relative team strength
       df['relative_strength'] = df['winning_percentage'] / df['league_strength']
       
       # Competition balance (standard deviation of wins in the year)
       df['competitive_balance'] = df.groupby('year')['wins'].transform('std')
       df['normalized_competitive_balance'] = (
           df['competitive_balance'] / df['total_games']
       )
       
       # Percentile rankings within year
       df['wins_percentile'] = df.groupby('year')['wins'].rank(pct=True)
       df['run_diff_percentile'] = df.groupby('year')['run_differential'].rank(pct=True)
       
       # Elite team indicators
       df['elite_team'] = (df['wins_percentile'] >= 0.9).astype(int)
       df['above_average_team'] = (df['winning_percentage'] > 0.5).astype(int)
       
       return df
   
    def _create_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
       """Create features measuring team consistency"""
       # Performance consistency (low variance is good)
       if 'performance_volatility' in df.columns:
           df['consistency_score'] = 1 / (1 + df['performance_volatility'])
       
       # Home/away balance (closer to 0 means more consistent)
       if 'home_away_diff' in df.columns:
           df['location_consistency'] = 1 - np.abs(df['home_away_diff'])
       
       # Run differential consistency
       df['run_diff_per_game'] = df['run_differential'] / df['total_games']
       df['run_diff_consistency'] = df.groupby('team_name')['run_diff_per_game'].transform(
           lambda x: 1 / (1 + x.rolling(window=5, min_periods=1).std())
       )
       
       # Expected vs actual performance consistency
       if 'pythagorean_win_pct' in df.columns:
           df['performance_reliability'] = 1 - np.abs(df['luck_pct'])
       
       return df
   
    def _create_era_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
       """Create era-adjusted features for historical comparisons"""
       # First ensure era column exists
       if 'era' not in df.columns:
           df = self._add_era_column(df)
       
       # Era-adjusted statistics (z-scores within era)
       era_adjusted_cols = ['wins', 'winning_percentage', 'run_differential', 
                           'runs_per_game', 'runs_allowed_per_game']
       
       for col in era_adjusted_cols:
           if col in df.columns:
               # Z-score within era
               df[f'{col}_era_zscore'] = df.groupby('era')[col].transform(
                   lambda x: (x - x.mean()) / x.std()
               )
               
               # Percentile within era
               df[f'{col}_era_percentile'] = df.groupby('era')[col].rank(pct=True)
       
       # Era-specific milestones
       era_milestones = {
           'dead_ball': {'elite_wins': 90, 'good_runs_scored': 600},
           'live_ball': {'elite_wins': 95, 'good_runs_scored': 700},
           'modern': {'elite_wins': 95, 'good_runs_scored': 800}
       }
       
       def get_era_milestone(row, metric):
           era = row['era']
           if era in era_milestones and metric in era_milestones[era]:
               return era_milestones[era][metric]
           return era_milestones['modern'][metric]  # Default to modern
       
       # Apply era-specific thresholds
       df['era_adjusted_elite_wins'] = df.apply(
           lambda row: row['wins'] >= get_era_milestone(row, 'elite_wins'), 
           axis=1
       ).astype(int)
       
       # Era offensive environment
       df['era_run_environment'] = df.groupby(['era', 'year'])['runs_per_game'].transform('mean')
       df['offense_vs_era'] = df['runs_per_game'] / df['era_run_environment']
       df['defense_vs_era'] = df['era_run_environment'] / df['runs_allowed_per_game']
       
       # Historical dominance score (how dominant was this team in its era)
       df['historical_dominance'] = (
           df['wins_era_percentile'] * 0.4 +
           df['run_differential_era_percentile'] * 0.3 +
           df['winning_percentage_era_percentile'] * 0.3
       )
       
       return df
   
    def _add_era_column(self, df: pd.DataFrame) -> pd.DataFrame:
       """Add era classification if not present"""
       def classify_era(year):
           if year < 1920:
               return 'dead_ball'
           elif year < 1942:
               return 'live_ball'
           elif year < 1963:
               return 'integration'
           elif year < 1977:
               return 'expansion'
           elif year < 1994:
               return 'free_agency'
           elif year < 2006:
               return 'steroid'
           else:
               return 'modern'
       
       df['era'] = df['year'].apply(classify_era)
       return df
   
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
       """Create target variables for different prediction tasks"""
       # Division winner (binary)
       df['is_division_winner'] = df['games_behind'].isna().astype(int)
       
       # Playoff team approximation (top teams by wins)
       # This is simplified - real playoff calculation would consider actual playoff rules
       year_team_counts = df.groupby('year')['team_name'].count()
       
       # Estimate playoff spots based on era (expanded over time)
       def get_playoff_spots(year):
           if year < 1969:
               return 2  # Only World Series teams
           elif year < 1995:
               return 4  # Division winners only
           elif year < 2012:
               return 8  # Added wild cards
           else:
               return 10  # Second wild card added
       
       df['playoff_spots'] = df['year'].apply(get_playoff_spots)
       df['made_playoffs'] = df.groupby('year').apply(
           lambda x: x.nlargest(x.iloc[0]['playoff_spots'], 'wins').index
       ).explode().isin(df.index).astype(int)
       
       # Milestone achievements
       df['achieved_90_wins'] = (df['wins'] >= 90).astype(int)
       df['achieved_95_wins'] = (df['wins'] >= 95).astype(int)
       df['achieved_100_wins'] = (df['wins'] >= 100).astype(int)
       
       # Run milestones
       df['scored_800_runs'] = (df['runs_scored'] >= 800).astype(int)
       df['allowed_under_650_runs'] = (df['runs_allowed'] < 650).astype(int)
       
       # Extreme outcomes
       df['terrible_season'] = (df['wins'] < 65).astype(int)
       df['historic_season'] = (df['wins'] >= 105).astype(int)
       
       # Future success indicator (requires look-ahead)
       df = df.sort_values(['team_name', 'year'])
       df['next_year_playoffs'] = df.groupby('team_name')['made_playoffs'].shift(-1)
       df['next_year_wins'] = df.groupby('team_name')['wins'].shift(-1)
       
       return df
   
    def _record_feature_names(self, df: pd.DataFrame):
       """Record the names of engineered features"""
       # Exclude original columns and target variables
       original_cols = [
           'team_name', 'year', 'wins', 'losses', 'winning_percentage',
           'games_behind', 'wild_card_games_behind', 'runs_scored', 
           'runs_allowed', 'run_differential', 'record_at_home',
           'record_when_away', 'record_against_top_50_percent',
           'record_in_the_last_10_games', 'current_streak',
           'expected_win_loss_record'
       ]
       
       target_cols = [
           'is_division_winner', 'made_playoffs', 'achieved_90_wins',
           'achieved_95_wins', 'achieved_100_wins', 'scored_800_runs',
           'allowed_under_650_runs', 'terrible_season', 'historic_season',
           'next_year_playoffs', 'next_year_wins'
       ]
       
       # Get all feature columns
       all_cols = set(df.columns)
       excluded_cols = set(original_cols + target_cols)
       
       self.feature_names = [col for col in df.columns if col not in excluded_cols]
       
       logger.info(f"Recorded {len(self.feature_names)} engineered features")
   
    def get_feature_groups(self) -> Dict[str, List[str]]:
       """Get features organized by category"""
       feature_groups = {
           'performance': [],
           'efficiency': [],
           'pythagorean': [],
           'historical': [],
           'trend': [],
           'strength': [],
           'consistency': [],
           'era_adjusted': []
       }
       
       # Categorize features based on naming patterns
       for feature in self.feature_names:
           if any(keyword in feature for keyword in ['home', 'away', 'vs_top']):
               feature_groups['performance'].append(feature)
           elif any(keyword in feature for keyword in ['runs_per', 'efficiency', 'rating']):
               feature_groups['efficiency'].append(feature)
           elif 'pythagorean' in feature or 'luck' in feature:
               feature_groups['pythagorean'].append(feature)
           elif any(keyword in feature for keyword in ['prev_', '_avg', '_change']):
               feature_groups['historical'].append(feature)
           elif any(keyword in feature for keyword in ['trend', 'momentum', 'streak']):
               feature_groups['trend'].append(feature)
           elif any(keyword in feature for keyword in ['strength', 'percentile', 'elite']):
               feature_groups['strength'].append(feature)
           elif 'consistency' in feature or 'volatility' in feature:
               feature_groups['consistency'].append(feature)
           elif 'era' in feature:
               feature_groups['era_adjusted'].append(feature)
       
       return feature_groups
   
    def select_features_for_model(self, model_type: str) -> List[str]:
       """Select appropriate features for different model types"""
       feature_groups = self.get_feature_groups()
       
       if model_type == 'division_winner':
           # Focus on current season performance and recent history
           selected = (
               feature_groups['performance'] +
               feature_groups['efficiency'] +
               feature_groups['pythagorean'] +
               [f for f in feature_groups['historical'] if 'prev_' in f] +
               feature_groups['trend'][:5]  # Top 5 trend features
           )
           
       elif model_type == 'win_total':
           # Include more historical and consistency features
           selected = (
               feature_groups['performance'] +
               feature_groups['efficiency'] +
               feature_groups['pythagorean'] +
               feature_groups['historical'] +
               feature_groups['consistency']
           )
           
       elif model_type == 'milestone':
           # Focus on strength and elite performance indicators
           selected = (
               feature_groups['performance'][:10] +
               feature_groups['efficiency'] +
               feature_groups['strength'] +
               feature_groups['era_adjusted'][:5]
           )
           
       else:
           # Default: use all features
           selected = self.feature_names
       
       # Remove any duplicates and ensure features exist
       selected = list(set(selected))
       selected = [f for f in selected if f in self.feature_names]
       
       logger.info(f"Selected {len(selected)} features for {model_type} model")
       
       return selected


# Example usage and testing
if __name__ == "__main__":
   logging.basicConfig(level=logging.INFO)
   
   # Create sample data
   sample_data = pd.DataFrame({
       'team_name': ['Yankees', 'Yankees', 'Red Sox', 'Red Sox'],
       'year': [2019, 2020, 2019, 2020],
       'wins': [103, 33, 84, 24],
       'losses': [59, 27, 78, 36],
       'runs_scored': [943, 315, 901, 292],
       'runs_allowed': [761, 281, 828, 351],
       'run_differential': [182, 34, 73, -59],
       'winning_percentage': [0.636, 0.550, 0.519, 0.400],
       'games_behind': [np.nan, 7.0, 19.0, 16.0],
       'at_home_wins': [57, 22, 43, 9],
       'at_home_losses': [24, 8, 38, 21],
       'when_away_wins': [46, 11, 41, 15],
       'when_away_losses': [35, 19, 40, 15]
   })
   
   # Initialize feature engineer
   engineer = FeatureEngineer(include_era_features=True)
   
   # Engineer features
   engineered_df = engineer.engineer_features(sample_data)
   
   print("Original shape:", sample_data.shape)
   print("Engineered shape:", engineered_df.shape)
   print("\nNew features created:", len(engineer.feature_names))
   
   # Show feature groups
   feature_groups = engineer.get_feature_groups()
   for group, features in feature_groups.items():
       if features:
           print(f"\n{group.upper()} features ({len(features)}):")
           print(f"  {', '.join(features[:5])}")
           if len(features) > 5:
               print(f"  ... and {len(features) - 5} more")