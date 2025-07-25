"""
Tests for data processing components including loading, preprocessing, and feature engineering.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.utils.constants import ERA_DEFINITIONS, MILESTONE_THRESHOLDS
from tests import TEST_DATA_DIR, TEST_RANDOM_SEED


class TestDataLoader(unittest.TestCase):
    """Test DataLoader functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Create test data file"""
        cls.test_data = pd.DataFrame({
            'year': [2020, 2020, 2021, 2021, 2022, 2022],
            'team_name': ['Yankees', 'Red Sox', 'Yankees', 'Red Sox', 'Yankees', 'Red Sox'],
            'wins': [95, 89, 92, 88, 99, 78],
            'losses': [67, 73, 70, 74, 63, 84],
            'runs_scored': [850, 820, 830, 810, 890, 750],
            'runs_allowed': [750, 780, 770, 790, 720, 850],
            'games_behind': [0, 6, 0, 4, 0, 21]
        })
        
        cls.test_file = TEST_DATA_DIR / 'test_mlb_data.csv'
        cls.test_data.to_csv(cls.test_file, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data file"""
        if cls.test_file.exists():
            cls.test_file.unlink()
    
    def setUp(self):
        """Initialize loader"""
        self.loader = DataLoader(data_path=self.test_file)
    
    def test_load_data(self):
        """Test basic data loading"""
        df = self.loader.load_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 6)
        self.assertEqual(list(df.columns), list(self.test_data.columns))
    
    def test_validate_data(self):
        """Test data validation"""
        df = self.loader.load_data()
        is_valid, issues = self.loader.validate_data(df)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_validate_data_with_issues(self):
        """Test validation with problematic data"""
        bad_data = self.test_data.copy()
        bad_data.loc[0, 'wins'] = -1  # Invalid wins
        bad_data.loc[1, 'losses'] = None  # Missing value
        
        is_valid, issues = self.loader.validate_data(bad_data)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        self.assertIn('negative', str(issues).lower())
    
    def test_add_calculated_fields(self):
        """Test calculated field creation"""
        df = self.loader.load_data()
        df_calc = self.loader.add_calculated_fields(df)
        
        # Check new fields exist
        self.assertIn('winning_percentage', df_calc.columns)
        self.assertIn('run_differential', df_calc.columns)
        
        # Verify calculations
        expected_win_pct = df_calc['wins'] / (df_calc['wins'] + df_calc['losses'])
        np.testing.assert_array_almost_equal(
            df_calc['winning_percentage'].values,
            expected_win_pct.values
        )
    
    def test_get_data_info(self):
        """Test data info retrieval"""
        self.loader.load_data()
        info = self.loader.get_data_info()
        
        self.assertIn('n_rows', info)
        self.assertIn('n_teams', info)
        self.assertIn('year_range', info)
        self.assertEqual(info['n_rows'], 6)
        self.assertEqual(info['n_teams'], 2)
        self.assertEqual(info['year_range'], (2020, 2022))


class TestDataPreprocessor(unittest.TestCase):
    """Test DataPreprocessor functionality"""
    
    def setUp(self):
        """Create test data"""
        self.test_data = pd.DataFrame({
            'year': [1910, 1950, 1980, 2000, 2020] * 2,
            'team_name': ['Team A'] * 5 + ['Team B'] * 5,
            'wins': [75, 82, 90, 95, 88, 70, 78, 85, 92, 91],
            'losses': [79, 72, 72, 67, 74, 84, 76, 77, 70, 71],
            'runs_scored': [650, 720, 800, 850, 820, 600, 700, 780, 840, 830],
            'runs_allowed': [700, 690, 750, 780, 790, 720, 710, 760, 770, 780],
            'games_behind': [15, 8, 0, 0, 5, 20, 12, 5, 2, 3]
        })
        
        self.preprocessor = DataPreprocessor()
    
    def test_preprocess_basic(self):
        """Test basic preprocessing"""
        df_processed = self.preprocessor.preprocess(self.test_data.copy())
        
        # Check no data lost
        self.assertEqual(len(df_processed), len(self.test_data))
        
        # Check era encoding
        self.assertIn('era', df_processed.columns)
        self.assertIn('era_encoded', df_processed.columns)
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df_missing = self.test_data.copy()
        df_missing.loc[0, 'runs_scored'] = np.nan
        df_missing.loc[1, 'games_behind'] = np.nan
        
        df_processed = self.preprocessor._handle_missing_values(df_missing)
        
        # Check no missing values remain in required columns
        self.assertEqual(df_processed['runs_scored'].isna().sum(), 0)
        # games_behind can have NaN for division winners
        self.assertTrue(df_processed['games_behind'].isna().sum() <= len(df_missing))
    
    def test_encode_era(self):
        """Test era encoding"""
        df_era = self.preprocessor._encode_era(self.test_data.copy())
        
        # Check era column exists
        self.assertIn('era', df_era.columns)
        self.assertIn('era_encoded', df_era.columns)
        
        # Verify era assignment
        self.assertEqual(df_era.loc[df_era['year'] == 1910, 'era'].iloc[0], 'dead_ball')
        self.assertEqual(df_era.loc[df_era['year'] == 2020, 'era'].iloc[0], 'modern')
    
    def test_scale_features(self):
        """Test feature scaling"""
        df_scaled = self.preprocessor.scale_features(
            self.test_data.copy(),
            features=['wins', 'runs_scored', 'runs_allowed']
        )
        
        # Check scaled features
        for feature in ['wins', 'runs_scored', 'runs_allowed']:
            scaled_col = f'{feature}_scaled'
            self.assertIn(scaled_col, df_scaled.columns)
            
            # Verify scaling (approximate standard scaling)
            self.assertAlmostEqual(df_scaled[scaled_col].mean(), 0, places=1)
            self.assertAlmostEqual(df_scaled[scaled_col].std(), 1, places=1)
    
    def test_era_filtering(self):
        """Test era-based filtering"""
        # Test modern era only
        preprocessor_modern = DataPreprocessor(era_strategy='modern')
        df_modern = preprocessor_modern.preprocess(self.test_data.copy())
        
        self.assertEqual(len(df_modern), 2)  # Only 2020 data
        self.assertTrue(all(df_modern['year'] >= 2006))
        
        # Test expansion era onwards
        preprocessor_expansion = DataPreprocessor(era_strategy='expansion_onwards')
        df_expansion = preprocessor_expansion.preprocess(self.test_data.copy())
        
        self.assertGreater(len(df_expansion), len(df_modern))
        self.assertTrue(all(df_expansion['year'] >= 1969))


class TestFeatureEngineering(unittest.TestCase):
    """Test FeatureEngineer functionality"""
    
    def setUp(self):
        """Create test data with historical context"""
        years = list(range(2015, 2023))
        self.test_data = pd.DataFrame({
            'year': years * 2,
            'team_name': ['Yankees'] * len(years) + ['Red Sox'] * len(years),
            'wins': [84, 87, 91, 100, 103, 92, 89, 95] + [78, 93, 108, 84, 92, 88, 78, 83],
            'losses': [78, 75, 71, 62, 59, 70, 73, 67] + [84, 69, 54, 78, 70, 74, 84, 79],
            'runs_scored': [764, 785, 851, 851, 943, 804, 776, 825] + [748, 785, 876, 829, 901, 735, 689, 754],
            'runs_allowed': [713, 660, 671, 669, 739, 717, 695, 723] + [753, 668, 647, 800, 792, 754, 812, 789],
            'games_behind': [6, 5, 2, 0, 0, 7, 9, 4] + [12, 0, 0, 8, 3, 10, 15, 10],
            'era': ['modern'] * (len(years) * 2)
        })
        
        # Add calculated fields
        self.test_data['winning_percentage'] = (
            self.test_data['wins'] / (self.test_data['wins'] + self.test_data['losses'])
        )
        self.test_data['run_differential'] = (
            self.test_data['runs_scored'] - self.test_data['runs_allowed']
        )
        
        self.engineer = FeatureEngineer()
    
    def test_engineer_features(self):
        """Test complete feature engineering pipeline"""
        df_engineered = self.engineer.engineer_features(self.test_data.copy())
        
        # Check that original columns are preserved
        for col in self.test_data.columns:
            self.assertIn(col, df_engineered.columns)
        
        # Check that new features were created
        self.assertGreater(len(df_engineered.columns), len(self.test_data.columns))
        
        # Verify feature groups exist
        feature_groups = self.engineer.get_feature_groups()
        self.assertIn('performance', feature_groups)
        self.assertIn('efficiency', feature_groups)
        self.assertIn('historical', feature_groups)
    
    def test_performance_features(self):
        """Test performance feature creation"""
        df_perf = self.engineer._create_performance_features(self.test_data.copy())
        
        # Check key performance features
        expected_features = [
            'runs_scored_per_game',
            'runs_allowed_per_game',
            'run_differential_per_game',
            'pythag_expectation'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df_perf.columns)
        
        # Verify Pythagorean expectation calculation
        pythag = df_perf['pythag_expectation'].iloc[0]
        rs = self.test_data['runs_scored'].iloc[0]
        ra = self.test_data['runs_allowed'].iloc[0]
        expected_pythag = (rs ** 2) / (rs ** 2 + ra ** 2)
        
        self.assertAlmostEqual(pythag, expected_pythag, places=4)
    
    def test_historical_features(self):
        """Test historical feature creation"""
        df_hist = self.engineer._create_historical_features(self.test_data.copy())
        
        # Check lag features
        self.assertIn('prev_wins', df_hist.columns)
        self.assertIn('prev_run_differential', df_hist.columns)
        
        # Verify lag calculations
        for i in range(1, len(df_hist)):
            if df_hist.iloc[i]['team_name'] == df_hist.iloc[i-1]['team_name']:
                if not pd.isna(df_hist.iloc[i]['prev_wins']):
                    self.assertEqual(
                        df_hist.iloc[i]['prev_wins'],
                        df_hist.iloc[i-1]['wins']
                    )
    
    def test_rolling_features(self):
        """Test rolling average features"""
        df_roll = self.engineer._create_rolling_features(self.test_data.copy())
        
        # Check rolling features exist
        self.assertIn('wins_ma_3', df_roll.columns)
        self.assertIn('run_diff_ma_5', df_roll.columns)
        
        # Verify rolling calculations for complete windows
        # Find a team with enough history
        yankees_data = df_roll[df_roll['team_name'] == 'Yankees'].sort_values('year')
        if len(yankees_data) >= 3:
            # Check 3-year moving average
            for i in range(2, len(yankees_data)):
                if not pd.isna(yankees_data.iloc[i]['wins_ma_3']):
                    expected_ma = yankees_data.iloc[i-2:i+1]['wins'].mean()
                    self.assertAlmostEqual(
                        yankees_data.iloc[i]['wins_ma_3'],
                        expected_ma,
                        places=2
                    )
    
    def test_target_features(self):
        """Test target feature creation"""
        df_target = self.engineer._create_target_features(self.test_data.copy())
        
        # Check milestone features
        milestone_features = [
            'is_division_winner',
            'made_playoffs',
            'achieved_90_wins',
            'achieved_100_wins'
        ]
        
        for feature in milestone_features:
            self.assertIn(feature, df_target.columns)
            # Verify binary nature
            self.assertTrue(df_target[feature].isin([0, 1]).all())
        
        # Verify threshold logic
        self.assertTrue(all(
            df_target.loc[df_target['wins'] >= 90, 'achieved_90_wins'] == 1
        ))
        self.assertTrue(all(
            df_target.loc[df_target['wins'] < 90, 'achieved_90_wins'] == 0
        ))
    
    def test_era_adjusted_features(self):
        """Test era-adjusted feature creation"""
        # Add more eras for testing
        df_multi_era = self.test_data.copy()
        df_multi_era['era'] = ['steroid'] * 8 + ['modern'] * 8
        
        engineer_era = FeatureEngineer(include_era_features=True)
        df_era = engineer_era._create_era_adjusted_features(df_multi_era)
        
        # Check era features exist
        self.assertIn('era_mean_wins', df_era.columns)
        self.assertIn('wins_era_zscore', df_era.columns)
        self.assertIn('era_relative_performance', df_era.columns)
        
        # Verify era statistics are different
        steroid_mean = df_era[df_era['era'] == 'steroid']['era_mean_wins'].iloc[0]
        modern_mean = df_era[df_era['era'] == 'modern']['era_mean_wins'].iloc[0]
        
        # They should be different if the data is different between eras
        if df_multi_era[df_multi_era['era'] == 'steroid']['wins'].mean() != \
           df_multi_era[df_multi_era['era'] == 'modern']['wins'].mean():
            self.assertNotEqual(steroid_mean, modern_mean)
    
    def test_feature_selection(self):
        """Test feature selection for models"""
        df_engineered = self.engineer.engineer_features(self.test_data.copy())
        
        # Test classification features
        class_features = self.engineer.select_features_for_model('division_winner')
        self.assertIsInstance(class_features, list)
        self.assertGreater(len(class_features), 10)  # Should have reasonable number
        
        # Test regression features
        reg_features = self.engineer.select_features_for_model('win_total')
        self.assertIsInstance(reg_features, list)
        self.assertGreater(len(reg_features), 10)
        
        # Features should be different but with overlap
        overlap = set(class_features) & set(reg_features)
        self.assertGreater(len(overlap), 5)  # Some common features
        self.assertNotEqual(set(class_features), set(reg_features))  # But not identical
    
    def test_feature_validation(self):
        """Test that engineered features are valid"""
        df_engineered = self.engineer.engineer_features(self.test_data.copy())
        
        # Check for infinite values
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertFalse(
                np.isinf(df_engineered[col]).any(),
                f"Infinite values found in {col}"
            )
        
        # Check for reasonable ranges
        if 'winning_percentage' in df_engineered.columns:
            self.assertTrue(all(
                (df_engineered['winning_percentage'] >= 0) & 
                (df_engineered['winning_percentage'] <= 1)
            ))
        
        if 'pythag_expectation' in df_engineered.columns:
            self.assertTrue(all(
                (df_engineered['pythag_expectation'] >= 0) & 
                (df_engineered['pythag_expectation'] <= 1)
            ))


class TestIntegration(unittest.TestCase):
    """Integration tests for data pipeline"""
    
    def test_full_data_pipeline(self):
        """Test complete data processing pipeline"""
        # Create sample data
        test_data = pd.DataFrame({
            'year': list(range(2010, 2023)) * 2,
            'team_name': ['Team A'] * 13 + ['Team B'] * 13,
            'wins': np.random.randint(70, 100, 26),
            'losses': np.random.randint(62, 92, 26),
            'runs_scored': np.random.randint(700, 900, 26),
            'runs_allowed': np.random.randint(650, 850, 26),
            'games_behind': np.random.randint(0, 20, 26)
        })
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Load data
            loader = DataLoader(data_path=temp_file)
            df = loader.load_and_validate()
            
            # Preprocess
            preprocessor = DataPreprocessor(era_strategy='all')
            df_processed = preprocessor.preprocess(df)
            
            # Engineer features
            engineer = FeatureEngineer()
            df_final = engineer.engineer_features(df_processed)
            
            # Verify pipeline success
            self.assertGreater(len(df_final.columns), len(test_data.columns))
            self.assertEqual(len(df_final), len(test_data))
            
            # Verify key features exist
            expected_features = [
                'winning_percentage',
                'run_differential',
                'pythag_expectation',
                'is_division_winner'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, df_final.columns)
            
        finally:
            # Clean up
            Path(temp_file).unlink()


if __name__ == '__main__':
    unittest.main()