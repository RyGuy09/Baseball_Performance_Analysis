"""
Tests for prediction components including:
- Prediction pipeline
- Individual predictors
- Confidence calculations
- Result formatting
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import json
from datetime import datetime

from src.prediction.predictor import DivisionWinnerPredictor, WinsPredictor, MilestonePredictor
from src.prediction.prediction_pipeline import PredictionPipeline
from src.data.feature_engineering import FeatureEngineer
from src.utils.helpers import set_random_seed
from tests import TEST_RANDOM_SEED, TEST_DATA_DIR


class TestDivisionWinnerPredictor(unittest.TestCase):
    """Test division winner prediction functionality"""
    
    def setUp(self):
        """Create test data and predictor"""
        set_random_seed(TEST_RANDOM_SEED)
        
        # Create synthetic team data
        self.test_data = pd.DataFrame({
            'team_name': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
            'wins': [95, 88, 81, 75, 68],
            'losses': [67, 74, 81, 87, 94],
            'run_differential': [120, 50, 0, -40, -100],
            'runs_scored': [850, 780, 720, 680, 620],
            'runs_allowed': [730, 730, 720, 720, 720],
            'winning_percentage': [0.586, 0.543, 0.500, 0.463, 0.420]
        })
        
        # Add more features
        engineer = FeatureEngineer()
        self.test_data = engineer.create_performance_features(self.test_data)
        
        self.predictor = DivisionWinnerPredictor()
    
    def test_predict_basic(self):
        """Test basic prediction functionality"""
        # Mock a trained model
        from sklearn.ensemble import RandomForestClassifier
        mock_model = RandomForestClassifier(n_estimators=10, random_state=TEST_RANDOM_SEED)
        
        # Create training data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        mock_model.fit(X_train, y_train)
        
        self.predictor.model = mock_model
        self.predictor.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Test prediction
        X_test = np.random.randn(5, 10)
        predictions = self.predictor.predict(X_test)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_predict_with_confidence(self):
        """Test prediction with confidence scores"""
        # Mock model
        from sklearn.ensemble import RandomForestClassifier
        mock_model = RandomForestClassifier(n_estimators=10, random_state=TEST_RANDOM_SEED)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        mock_model.fit(X_train, y_train)
        
        self.predictor.model = mock_model
        self.predictor.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Test prediction with confidence
        X_test = np.random.randn(5, 10)
        results = self.predictor.predict_with_confidence(X_test)
        
        self.assertIn('predictions', results)
        self.assertIn('probabilities', results)
        self.assertIn('confidence_scores', results)
        self.assertIn('confidence_levels', results)
        
        # Check confidence levels
        for level in results['confidence_levels']:
            self.assertIn(level, ['Low', 'Medium', 'High', 'Very High'])
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        # Test various probability values
        test_probas = [
            [0.5, 0.5],    # Low confidence
            [0.7, 0.3],    # Medium confidence
            [0.85, 0.15],  # High confidence
            [0.95, 0.05],  # Very high confidence
        ]
        
        for proba in test_probas:
            confidence = self.predictor._calculate_confidence(proba)
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
            
            level = self.predictor._get_confidence_level(confidence)
            self.assertIn(level, ['Low', 'Medium', 'High', 'Very High'])


class TestWinsPredictor(unittest.TestCase):
    """Test wins prediction functionality"""
    
    def setUp(self):
        """Create test data and predictor"""
        set_random_seed(TEST_RANDOM_SEED)
        
        self.test_data = pd.DataFrame({
            'team_name': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
            'prev_wins': [92, 85, 79, 73, 66],
            'prev_run_differential': [100, 30, -10, -50, -120],
            'runs_scored_per_game': [5.2, 4.8, 4.5, 4.2, 3.8],
            'runs_allowed_per_game': [4.5, 4.6, 4.5, 4.6, 5.0],
            'pythag_expectation': [0.580, 0.520, 0.500, 0.460, 0.400]
        })
        
        self.predictor = WinsPredictor()
    
    def test_predict_basic(self):
        """Test basic win prediction"""
        # Mock model
        from sklearn.ensemble import RandomForestRegressor
        mock_model = RandomForestRegressor(n_estimators=10, random_state=TEST_RANDOM_SEED)
        X_train = np.random.randn(100, 10)
        y_train = np.random.uniform(60, 100, 100)  # Win range
        mock_model.fit(X_train, y_train)
        
        self.predictor.model = mock_model
        self.predictor.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Test prediction
        X_test = np.random.randn(5, 10)
        predictions = self.predictor.predict(X_test)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(40 <= p <= 120 for p in predictions))  # Reasonable range
    
    def test_predict_with_bounds(self):
        """Test prediction with confidence bounds"""
        # Mock model with bounds capability
        from sklearn.ensemble import RandomForestRegressor
        mock_model = RandomForestRegressor(n_estimators=50, random_state=TEST_RANDOM_SEED)
        X_train = np.random.randn(200, 10)
        y_train = 81 + np.random.randn(200) * 10  # Mean 81, std 10
        mock_model.fit(X_train, y_train)
        
        self.predictor.model = mock_model
        self.predictor.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Test prediction with bounds
        X_test = np.random.randn(5, 10)
        results = self.predictor.predict_with_bounds(X_test)
        
        self.assertIn('predictions', results)
        self.assertIn('lower_bounds', results)
        self.assertIn('upper_bounds', results)
        self.assertIn('confidence_level', results)
        
        # Check bounds make sense
        for i in range(len(results['predictions'])):
            self.assertLess(results['lower_bounds'][i], results['predictions'][i])
            self.assertGreater(results['upper_bounds'][i], results['predictions'][i])
            
            # Check reasonable interval width (not too narrow or wide)
            interval_width = results['upper_bounds'][i] - results['lower_bounds'][i]
            self.assertGreater(interval_width, 5)   # At least 5 wins
            self.assertLess(interval_width, 30)     # At most 30 wins
    
    def test_quantile_prediction(self):
        """Test quantile-based prediction bounds"""
        # Create predictor with quantile method
        predictor = WinsPredictor(prediction_method='quantile')
        
        # Mock quantile model
        from sklearn.ensemble import GradientBoostingRegressor
        lower_model = GradientBoostingRegressor(
            loss='quantile', alpha=0.025, n_estimators=10, random_state=TEST_RANDOM_SEED
        )
        upper_model = GradientBoostingRegressor(
            loss='quantile', alpha=0.975, n_estimators=10, random_state=TEST_RANDOM_SEED
        )
        
        X_train = np.random.randn(200, 10)
        y_train = 81 + np.random.randn(200) * 10
        
        lower_model.fit(X_train, y_train)
        upper_model.fit(X_train, y_train)
        
        predictor.lower_model = lower_model
        predictor.upper_model = upper_model
        predictor.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Test bounds
        X_test = np.random.randn(5, 10)
        bounds = predictor._predict_quantile_bounds(X_test)
        
        self.assertIn('lower', bounds)
        self.assertIn('upper', bounds)
        self.assertTrue(all(bounds['lower'] < bounds['upper']))


class TestMilestonePredictor(unittest.TestCase):
    """Test milestone prediction functionality"""
    
    def setUp(self):
        """Create test data and predictor"""
        set_random_seed(TEST_RANDOM_SEED)
        
        self.test_data = pd.DataFrame({
            'team_name': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
            'wins': [95, 88, 81, 75, 68],
            'runs_scored': [850, 780, 720, 680, 620],
            'runs_allowed': [730, 730, 720, 720, 720],
            'pythag_expectation': [0.580, 0.520, 0.500, 0.460, 0.400]
        })
        
        self.predictor = MilestonePredictor()
        self.milestone_names = ['90_wins', '100_wins', 'playoffs']
    
    def test_predict_milestones(self):
        """Test milestone prediction"""
        # Mock multi-output model
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        base_model = RandomForestClassifier(n_estimators=10, random_state=TEST_RANDOM_SEED)
        mock_model = MultiOutputClassifier(base_model)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, (100, 3))  # 3 milestones
        mock_model.fit(X_train, y_train)
        
        self.predictor.model = mock_model
        self.predictor.milestone_names = self.milestone_names
        self.predictor.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Test prediction
        X_test = np.random.randn(5, 10)
        predictions = self.predictor.predict(X_test)
        
        self.assertEqual(predictions.shape, (5, 3))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
    
    def test_predict_proba_milestones(self):
        """Test milestone probability prediction"""
        # Mock model
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        base_model = RandomForestClassifier(n_estimators=10, random_state=TEST_RANDOM_SEED)
        mock_model = MultiOutputClassifier(base_model)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, (100, 3))
        mock_model.fit(X_train, y_train)
        
        self.predictor.model = mock_model
        self.predictor.milestone_names = self.milestone_names
        self.predictor.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Test probability prediction
        X_test = np.random.randn(5, 10)
        probas = self.predictor.predict_proba(X_test)
        
        self.assertIsInstance(probas, dict)
        self.assertEqual(len(probas), 3)
        
        for milestone in self.milestone_names:
            self.assertIn(milestone, probas)
            self.assertEqual(len(probas[milestone]), 5)
            self.assertTrue(np.all((probas[milestone] >= 0) & (probas[milestone] <= 1)))


class TestPredictionPipeline(unittest.TestCase):
    """Test complete prediction pipeline"""
    
    def setUp(self):
        """Create test environment"""
        set_random_seed(TEST_RANDOM_SEED)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'team_name': [f'Team {i}' for i in range(30)],
            'year': [2024] * 30,
            'wins': np.random.randint(65, 100, 30),
            'losses': np.random.randint(62, 97, 30),
            'runs_scored': np.random.randint(650, 900, 30),
            'runs_allowed': np.random.randint(650, 900, 30),
            'games_behind': np.random.randint(0, 25, 30)
        })
        
        # Calculate derived features
        self.test_data['winning_percentage'] = (
            self.test_data['wins'] / (self.test_data['wins'] + self.test_data['losses'])
        )
        self.test_data['run_differential'] = (
            self.test_data['runs_scored'] - self.test_data['runs_allowed']
        )
        
        self.pipeline = PredictionPipeline()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsInstance(self.pipeline, PredictionPipeline)
        self.assertIsNone(self.pipeline.division_predictor)
        self.assertIsNone(self.pipeline.wins_predictor)
        self.assertIsNone(self.pipeline.milestone_predictor)
    
    def test_load_models(self):
        """Test model loading"""
        # Create temporary model files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create mock model files
            models_dir = tmp_path / 'models' / 'saved_models'
            scalers_dir = tmp_path / 'models' / 'scalers'
            models_dir.mkdir(parents=True)
            scalers_dir.mkdir(parents=True)
            
            # Save mock models
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            import joblib
            
            # Division classifier
            div_model = RandomForestClassifier(n_estimators=10)
            div_model.fit(np.random.randn(100, 10), np.random.randint(0, 2, 100))
            joblib.dump(div_model, models_dir / 'division_classifier.pkl')
            
            # Wins regressor
            wins_model = RandomForestRegressor(n_estimators=10)
            wins_model.fit(np.random.randn(100, 10), np.random.uniform(60, 100, 100))
            joblib.dump(wins_model, models_dir / 'wins_regressor.pkl')
            
            # Scalers
            scaler = StandardScaler()
            scaler.fit(np.random.randn(100, 10))
            joblib.dump(scaler, scalers_dir / 'classification_scaler.pkl')
            joblib.dump(scaler, scalers_dir / 'regression_scaler.pkl')
            
            # Update pipeline paths
            self.pipeline.model_dir = models_dir
            self.pipeline.scaler_dir = scalers_dir
            
            # Test loading
            self.pipeline.load_models()
            
            self.assertIsNotNone(self.pipeline.division_predictor.model)
            self.assertIsNotNone(self.pipeline.wins_predictor.model)
    
    def test_predict_season(self):
        """Test full season prediction"""
        # Mock predictors
        self.pipeline.division_predictor = DivisionWinnerPredictor()
        self.pipeline.wins_predictor = WinsPredictor()
        self.pipeline.milestone_predictor = MilestonePredictor()
        
        # Add feature engineering
        engineer = FeatureEngineer()
        test_data_with_features = engineer.engineer_features(self.test_data)
        
        # Mock models for predictors
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.multioutput import MultiOutputClassifier
        
        # Division model
        div_model = RandomForestClassifier(n_estimators=10, random_state=TEST_RANDOM_SEED)
        n_features = 20
        div_model.fit(
            np.random.randn(100, n_features), 
            np.random.randint(0, 2, 100)
        )
        self.pipeline.division_predictor.model = div_model
        self.pipeline.division_predictor.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Wins model
        wins_model = RandomForestRegressor(n_estimators=10, random_state=TEST_RANDOM_SEED)
        wins_model.fit(
            np.random.randn(100, n_features),
            np.random.uniform(60, 100, 100)
        )
        self.pipeline.wins_predictor.model = wins_model
        self.pipeline.wins_predictor.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Milestone model
        base_model = RandomForestClassifier(n_estimators=10, random_state=TEST_RANDOM_SEED)
        milestone_model = MultiOutputClassifier(base_model)
        milestone_model.fit(
            np.random.randn(100, n_features),
            np.random.randint(0, 2, (100, 3))
        )
        self.pipeline.milestone_predictor.model = milestone_model
        self.pipeline.milestone_predictor.milestone_names = ['90_wins', '100_wins', 'playoffs']
        self.pipeline.milestone_predictor.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create mock features
        mock_features = pd.DataFrame(
            np.random.randn(30, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        test_data_with_mock_features = pd.concat([
            test_data_with_features[['team_name', 'year']], 
            mock_features
        ], axis=1)
        
        # Test prediction
        predictions = self.pipeline.predict_season(
            test_data_with_mock_features,
            include_confidence=True,
            include_milestones=True
        )
        
        # Check results
        self.assertEqual(len(predictions), 30)
        self.assertIn('team_name', predictions.columns)
        self.assertIn('predicted_wins', predictions.columns)
        self.assertIn('division_winner_prediction', predictions.columns)
        self.assertIn('division_winner_probability', predictions.columns)
        self.assertIn('division_winner_confidence', predictions.columns)
        
        # Check milestones if included
        self.assertIn('prob_90_wins', predictions.columns)
        self.assertIn('prob_100_wins', predictions.columns)
        self.assertIn('prob_playoffs', predictions.columns)
    
    def test_format_results(self):
        """Test result formatting"""
        # Create mock predictions
        raw_predictions = {
            'team_name': ['Team A', 'Team B', 'Team C'],
            'predicted_wins': [95.2, 87.8, 73.1],
            'win_lower': [89.5, 82.1, 68.2],
            'win_upper': [100.9, 93.5, 78.0],
            'division_prediction': [1, 0, 0],
            'division_proba': [0.82, 0.31, 0.15],
            'division_confidence': [0.82, 0.62, 0.70],
            'milestone_90_wins': [0.75, 0.15, 0.02],
            'milestone_100_wins': [0.25, 0.02, 0.00],
            'milestone_playoffs': [0.85, 0.35, 0.10]
        }
        
        # Test formatting
        formatted = self.pipeline._format_results(raw_predictions)
        
        self.assertIsInstance(formatted, pd.DataFrame)
        self.assertIn('predicted_wins', formatted.columns)
        self.assertIn('division_winner_probability', formatted.columns)
        
        # Check data types and ranges
        self.assertTrue(all(0 <= p <= 1 for p in formatted['division_winner_probability']))
        self.assertTrue(all(40 <= w <= 120 for w in formatted['predicted_wins']))
    
    def test_save_predictions(self):
        """Test saving predictions"""
        # Create mock predictions
        predictions = pd.DataFrame({
            'team_name': [f'Team {i}' for i in range(30)],
            'predicted_wins': np.random.uniform(65, 95, 30),
            'division_winner_probability': np.random.uniform(0, 1, 30),
            'division_winner_confidence': np.random.choice(['Low', 'Medium', 'High'], 30)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / 'predictions.csv'
            
            # Save predictions
            self.pipeline.save_predictions(predictions, output_path)
            
            # Check file exists
            self.assertTrue(output_path.exists())
            
            # Load and verify
            loaded_predictions = pd.read_csv(output_path)
            self.assertEqual(len(loaded_predictions), 30)
            self.assertListEqual(
                list(loaded_predictions.columns),
                list(predictions.columns)
            )
    
    def test_generate_report(self):
        """Test report generation"""
        # Create mock predictions
        predictions = pd.DataFrame({
            'team_name': [f'Team {i}' for i in range(30)],
            'predicted_wins': np.random.uniform(65, 95, 30),
            'division_winner_probability': np.random.uniform(0, 1, 30),
            'division_winner_prediction': np.random.randint(0, 2, 30),
            'prob_90_wins': np.random.uniform(0, 0.5, 30),
            'prob_100_wins': np.random.uniform(0, 0.1, 30)
        })
        
        # Generate report
        report = self.pipeline.generate_prediction_report(predictions)
        
        self.assertIsInstance(report, dict)
        self.assertIn('summary', report)
        self.assertIn('top_teams', report)
        self.assertIn('division_favorites', report)
        self.assertIn('milestone_analysis', report)
        self.assertIn('timestamp', report)
        
        # Check summary statistics
        summary = report['summary']
        self.assertIn('total_teams', summary)
        self.assertIn('avg_predicted_wins', summary)
        self.assertIn('predicted_division_winners', summary)
        
        # Check timestamp format
        timestamp = datetime.fromisoformat(report['timestamp'])
        self.assertIsInstance(timestamp, datetime)


class TestPredictionIntegration(unittest.TestCase):
    """Integration tests for complete prediction system"""
    
    def test_end_to_end_prediction(self):
        """Test complete prediction workflow"""
        set_random_seed(TEST_RANDOM_SEED)
        
        # Create realistic test data
        team_names = [
            'Yankees', 'Red Sox', 'Rays', 'Orioles', 'Blue Jays',
            'Twins', 'White Sox', 'Guardians', 'Tigers', 'Royals',
            'Astros', 'Rangers', 'Mariners', 'Athletics', 'Angels',
            'Braves', 'Phillies', 'Mets', 'Marlins', 'Nationals',
            'Brewers', 'Cardinals', 'Cubs', 'Reds', 'Pirates',
            'Dodgers', 'Padres', 'Giants', 'Diamondbacks', 'Rockies'
        ]
        
        test_data = pd.DataFrame({
            'team_name': team_names,
            'year': [2024] * 30,
            'wins': np.random.normal(81, 10, 30).astype(int).clip(60, 105),
            'losses': [162 - w for w in np.random.normal(81, 10, 30).astype(int).clip(60, 105)],
            'runs_scored': np.random.normal(750, 75, 30).astype(int),
            'runs_allowed': np.random.normal(750, 75, 30).astype(int)
        })
        
        # Calculate derived features
        test_data['winning_percentage'] = test_data['wins'] / 162
        test_data['run_differential'] = test_data['runs_scored'] - test_data['runs_allowed']
        
        # Add previous year data
        test_data['prev_wins'] = test_data['wins'] + np.random.randint(-10, 10, 30)
        test_data['prev_run_differential'] = test_data['run_differential'] + np.random.randint(-50, 50, 30)
        
        # Engineer features
        engineer = FeatureEngineer()
        test_data_featured = engineer.engineer_features(test_data)
        
        # Create pipeline
        pipeline = PredictionPipeline()
        
        # Mock predictors with simple models
        from sklearn.dummy import DummyClassifier, DummyRegressor
        
        # Division predictor
        pipeline.division_predictor = DivisionWinnerPredictor()
        pipeline.division_predictor.model = DummyClassifier(strategy='stratified')
        pipeline.division_predictor.model.fit(
            np.random.randn(100, 10),
            np.random.randint(0, 2, 100)
        )
        pipeline.division_predictor.feature_names = test_data_featured.columns[:10].tolist()
        
        # Wins predictor
        pipeline.wins_predictor = WinsPredictor()
        pipeline.wins_predictor.model = DummyRegressor(strategy='mean')
        pipeline.wins_predictor.model.fit(
            np.random.randn(100, 10),
            np.random.uniform(65, 95, 100)
        )
        pipeline.wins_predictor.feature_names = test_data_featured.columns[:10].tolist()
        
        # Make predictions
        predictions = pipeline.predict_season(
            test_data_featured,
            include_confidence=True,
            include_milestones=False  # Skip milestones for simplicity
        )
        
        # Verify predictions
        self.assertEqual(len(predictions), 30)
        self.assertTrue(all(40 <= w <= 120 for w in predictions['predicted_wins']))
        self.assertTrue(all(0 <= p <= 1 for p in predictions['division_winner_probability']))
        
        # Generate report
        report = pipeline.generate_prediction_report(predictions)
        self.assertIsInstance(report, dict)
        self.assertIn('summary', report)
        
        # Test saving
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / 'test_predictions.csv'
            pipeline.save_predictions(predictions, output_path)
            self.assertTrue(output_path.exists())
            
            # Also test JSON format
            json_path = Path(tmp_dir) / 'test_predictions.json'
            predictions.to_json(json_path, orient='records', indent=2)
            self.assertTrue(json_path.exists())
            
            # Verify JSON is valid
            with open(json_path, 'r') as f:
                loaded_json = json.load(f)
            self.assertEqual(len(loaded_json), 30)


if __name__ == '__main__':
    unittest.main()