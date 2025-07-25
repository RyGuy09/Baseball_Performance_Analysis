"""
Tests for model components including:
- Model training
- Model evaluation
- Hyperparameter tuning
- Ensemble models
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import tempfile
from pathlib import Path

from src.models.classification_models import BaseballClassifier
from src.models.regression_models import BaseballRegressor
from src.models.ensemble_models import MLBEnsembleModel
from src.models.milestone_predictor import MilestonePredictor
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.evaluation.metrics import ClassificationMetrics, RegressionMetrics
from tests import TEST_RANDOM_SEED


class TestClassificationModels(unittest.TestCase):
    """Test classification model functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic classification data"""
        np.random.seed(TEST_RANDOM_SEED)
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=TEST_RANDOM_SEED
        )
        
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=TEST_RANDOM_SEED
        )
        
        # Create feature names
        cls.feature_names = [f'feature_{i}' for i in range(20)]
    
    def test_logistic_regression(self):
        """Test logistic regression model"""
        model = BaseballClassifier(model_type='logistic')
        model.train(self.X_train, self.y_train)
        
        # Test predictions
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Test probabilities
        probas = model.predict_proba(self.X_test)
        self.assertEqual(probas.shape, (len(self.y_test), 2))
        self.assertTrue(np.allclose(probas.sum(axis=1), 1.0))
        
        # Test performance
        metrics = ClassificationMetrics(self.y_test, predictions, probas)
        metrics_dict = metrics.get_metrics()
        self.assertGreater(metrics_dict['accuracy'], 0.6)
    
    def test_random_forest_classifier(self):
        """Test random forest classifier"""
        model = BaseballClassifier(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        probas = model.predict_proba(self.X_test)
        
        # Test feature importance
        importance = model.get_feature_importance(self.feature_names)
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), len(self.feature_names))
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
    
    def test_xgboost_classifier(self):
        """Test XGBoost classifier"""
        model = BaseballClassifier(model_type='xgboost')
        model.train(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Test with validation set
        X_val = self.X_test[:50]
        y_val = self.y_test[:50]
        
        model = BaseballClassifier(model_type='xgboost')
        model.train(
            self.X_train, 
            self.y_train,
            X_val=X_val,
            y_val=y_val
        )
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / 'test_model.pkl'
            
            # Train and save model
            model = BaseballClassifier(model_type='random_forest')
            model.train(self.X_train, self.y_train)
            original_predictions = model.predict(self.X_test)
            
            model.save_model(model_path)
            self.assertTrue(model_path.exists())
            
            # Load and test model
            loaded_model = BaseballClassifier(model_type='random_forest')
            loaded_model.load_model(model_path)
            loaded_predictions = loaded_model.predict(self.X_test)
            
            np.testing.assert_array_equal(original_predictions, loaded_predictions)


class TestRegressionModels(unittest.TestCase):
    """Test regression model functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic regression data"""
        np.random.seed(TEST_RANDOM_SEED)
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            noise=10,
            random_state=TEST_RANDOM_SEED
        )
        
        # Scale target to realistic win range
        y = 81 + (y - y.mean()) / y.std() * 10  # Mean 81 wins, std 10
        
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=TEST_RANDOM_SEED
        )
        
        cls.feature_names = [f'feature_{i}' for i in range(20)]
    
    def test_ridge_regression(self):
        """Test ridge regression model"""
        model = BaseballRegressor(model_type='ridge')
        model.train(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Test performance
        metrics = RegressionMetrics(self.y_test, predictions)
        metrics_dict = metrics.metrics
        self.assertLess(metrics_dict['rmse'], 15)  # Reasonable for win predictions
    
    def test_random_forest_regressor(self):
        """Test random forest regressor"""
        model = BaseballRegressor(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        
        # Test prediction bounds
        bounds = model.predict_with_bounds(self.X_test, confidence=0.95)
        self.assertIn('predictions', bounds)
        self.assertIn('lower_bound', bounds)
        self.assertIn('upper_bound', bounds)
        
        # Verify bounds contain true values most of the time
        in_bounds = (
            (self.y_test >= bounds['lower_bound']) & 
            (self.y_test <= bounds['upper_bound'])
        ).mean()
        self.assertGreater(in_bounds, 0.8)  # Should be ~95% but allow some margin
    
    def test_lightgbm_regressor(self):
        """Test LightGBM regressor"""
        model = BaseballRegressor(model_type='lightgbm')
        model.train(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Test feature importance
        importance = model.get_feature_importance(self.feature_names)
        self.assertIsNotNone(importance)
        self.assertTrue(all(importance['importance'] >= 0))


class TestEnsembleModels(unittest.TestCase):
    """Test ensemble model functionality"""
    
    def setUp(self):
        """Create test data and base models"""
        np.random.seed(TEST_RANDOM_SEED)
        
        # Classification data
        X_class, y_class = make_classification(
            n_samples=500, n_features=10, random_state=TEST_RANDOM_SEED
        )
        self.X_train_c, self.X_test_c, self.y_train_c, self.y_test_c = train_test_split(
            X_class, y_class, test_size=0.2, random_state=TEST_RANDOM_SEED
        )
        
        # Regression data
        X_reg, y_reg = make_regression(
            n_samples=500, n_features=10, random_state=TEST_RANDOM_SEED
        )
        self.X_train_r, self.X_test_r, self.y_train_r, self.y_test_r = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=TEST_RANDOM_SEED
        )
        
        # Create base models
        self.base_classifiers = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=TEST_RANDOM_SEED)),
            ('rf2', RandomForestClassifier(n_estimators=20, random_state=TEST_RANDOM_SEED+1))
        ]
        
        self.base_regressors = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=TEST_RANDOM_SEED)),
            ('rf2', RandomForestRegressor(n_estimators=20, random_state=TEST_RANDOM_SEED+1))
        ]
    
    def test_voting_classifier_ensemble(self):
        """Test voting classifier ensemble"""
        ensemble = MLBEnsembleModel(
            ensemble_type='voting',
            task_type='classification',
            models=self.base_classifiers
        )
        
        ensemble.train(self.X_train_c, self.y_train_c)
        predictions = ensemble.predict(self.X_test_c)
        
        self.assertEqual(len(predictions), len(self.y_test_c))
        
        # Test probabilities
        probas = ensemble.predict_proba(self.X_test_c)
        self.assertEqual(probas.shape[1], 2)
    
    def test_stacking_regressor_ensemble(self):
        """Test stacking regressor ensemble"""
        ensemble = MLBEnsembleModel(
            ensemble_type='stacking',
            task_type='regression',
            models=self.base_regressors
        )
        
        # Need validation set for stacking
        X_val = self.X_test_r[:20]
        y_val = self.y_test_r[:20]
        
        ensemble.train(
            self.X_train_r, 
            self.y_train_r,
            X_val=X_val,
            y_val=y_val
        )
        
        predictions = ensemble.predict(self.X_test_r[20:])
        self.assertEqual(len(predictions), len(self.y_test_r[20:]))
    
    def test_blending_ensemble(self):
        """Test blending ensemble"""
        ensemble = MLBEnsembleModel(
            ensemble_type='blending',
            task_type='classification',
            models=self.base_classifiers,
            blend_features=5
        )
        
        X_val = self.X_test_c[:20]
        y_val = self.y_test_c[:20]
        
        ensemble.train(
            self.X_train_c,
            self.y_train_c,
            X_val=X_val,
            y_val=y_val
        )
        
        predictions = ensemble.predict(self.X_test_c[20:])
        self.assertEqual(len(predictions), len(self.y_test_c[20:]))


class TestMilestonePredictor(unittest.TestCase):
    """Test milestone predictor functionality"""
    
    def setUp(self):
        """Create test data for milestone prediction"""
        np.random.seed(TEST_RANDOM_SEED)
        
        # Create synthetic data with multiple targets
        n_samples = 500
        n_features = 15
        
        X = np.random.randn(n_samples, n_features)
        
        # Create correlated milestone targets
        y_90_wins = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 1).astype(int)
        y_100_wins = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 2).astype(int)
        y_playoffs = (X[:, 2] + X[:, 3] + np.random.randn(n_samples) * 0.5 > 0.5).astype(int)
        
        self.X = X
        self.y = np.column_stack([y_90_wins, y_100_wins, y_playoffs])
        self.milestone_names = ['90_wins', '100_wins', 'playoffs']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=TEST_RANDOM_SEED
        )
    
    def test_milestone_predictor_training(self):
        """Test milestone predictor training"""
        predictor = MilestonePredictor()
        predictor.train(
            self.X_train,
            self.y_train,
            milestone_names=self.milestone_names
        )
        
        # Test predictions
        predictions = predictor.predict(self.X_test)
        self.assertEqual(predictions.shape, self.y_test.shape)
        
        # Test probabilities
        probas = predictor.predict_proba(self.X_test)
        self.assertEqual(len(probas), len(self.milestone_names))
        
        for proba in probas.values():
            self.assertEqual(proba.shape[0], len(self.y_test))
            self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
    
    def test_milestone_calibration(self):
        """Test probability calibration"""
        predictor = MilestonePredictor(calibrate_probabilities=True)
        
        # Need validation set for calibration
        X_val = self.X_test[:50]
        y_val = self.y_test[:50]
        
        predictor.train(
            self.X_train,
            self.y_train,
            milestone_names=self.milestone_names,
            X_val=X_val,
            y_val=y_val
        )
        
        probas = predictor.predict_proba(self.X_test[50:])
        
        # Check calibration improved
        for milestone_idx, milestone in enumerate(self.milestone_names):
            proba = probas[milestone]
            true_labels = self.y_test[50:, milestone_idx]
            
            # Rough calibration check
            for threshold in [0.2, 0.5, 0.8]:
                predicted_positive_rate = (proba > threshold).mean()
                actual_positive_rate = true_labels[proba > threshold].mean() if (proba > threshold).any() else 0
                
                # Allow some tolerance
                if predicted_positive_rate > 0:
                    self.assertLess(
                        abs(predicted_positive_rate - actual_positive_rate),
                        0.3  # Rough tolerance for small test set
                    )


class TestHyperparameterTuning(unittest.TestCase):
    """Test hyperparameter tuning functionality"""
    
    def setUp(self):
        """Create test data"""
        np.random.seed(TEST_RANDOM_SEED)
        
        # Simple dataset for fast testing
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            random_state=TEST_RANDOM_SEED
        )
        
        self.X = X
        self.y = y
    
    def test_grid_search_tuning(self):
        """Test grid search hyperparameter tuning"""
        tuner = HyperparameterTuner(
            model_class='classification',
            model_type='random_forest',
            search_method='grid',
            cv_folds=3
        )
        
        # Define small parameter grid for testing
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        results = tuner.optimize(
            self.X,
            self.y,
            param_space=param_grid
        )
        
        self.assertIn('best_params', results)
        self.assertIn('best_score', results)
        self.assertIn('cv_results', results)
        
        # Check best params are from grid
        for param, value in results['best_params'].items():
            self.assertIn(value, param_grid[param])
    
    def test_random_search_tuning(self):
        """Test random search hyperparameter tuning"""
        tuner = HyperparameterTuner(
            model_class='classification',
            model_type='xgboost',
            search_method='random',
            n_iter=5,  # Small number for testing
            cv_folds=2
        )
        
        results = tuner.optimize(self.X, self.y)
        
        self.assertIn('best_params', results)
        self.assertIn('best_score', results)
        self.assertIsInstance(results['best_score'], float)
        
        # Test that model can be created with best params
        model = BaseballClassifier(
            model_type='xgboost',
            **results['best_params']
        )
        model.train(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_custom_scoring(self):
        """Test custom scoring function"""
        from sklearn.metrics import make_scorer, f1_score
        
        custom_scorer = make_scorer(f1_score, average='binary')
        
        tuner = HyperparameterTuner(
            model_class='classification',
            model_type='logistic',
            search_method='grid',
            cv_folds=3,
            scoring=custom_scorer
        )
        
        param_grid = {
            'C': [0.1, 1.0],
            'penalty': ['l2']
        }
        
        results = tuner.optimize(
            self.X,
            self.y,
            param_space=param_grid
        )
        
        self.assertIn('best_score', results)
        self.assertGreater(results['best_score'], 0)  # F1 score should be positive


class TestModelIntegration(unittest.TestCase):
    """Integration tests for complete model pipeline"""
    
    def test_classification_pipeline(self):
        """Test complete classification pipeline"""
        # Create data
        np.random.seed(TEST_RANDOM_SEED)
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            random_state=TEST_RANDOM_SEED
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=TEST_RANDOM_SEED
        )
        
        # Train multiple models
        models = {}
        for model_type in ['logistic', 'random_forest', 'xgboost']:
            model = BaseballClassifier(model_type=model_type)
            model.train(X_train, y_train)
            models[model_type] = model
        
        # Create ensemble
        model_list = [(name, model.model) for name, model in models.items()]
        ensemble = MLBEnsembleModel(
            ensemble_type='voting',
            task_type='classification',
            models=model_list
        )
        ensemble.train(X_train, y_train)
        
        # Evaluate all models
        results = {}
        for name, model in models.items():
            predictions = model.predict(X_test)
            probas = model.predict_proba(X_test)
            metrics = ClassificationMetrics(y_test, predictions, probas)
            results[name] = metrics.get_metrics()
        
        # Evaluate ensemble
        ensemble_pred = ensemble.predict(X_test)
        ensemble_proba = ensemble.predict_proba(X_test)
        ensemble_metrics = ClassificationMetrics(y_test, ensemble_pred, ensemble_proba)
        results['ensemble'] = ensemble_metrics.get_metrics()
        
        # Ensemble should perform at least as well as worst individual model
        min_accuracy = min(r['accuracy'] for r in results.values() if r != results['ensemble'])
        self.assertGreaterEqual(results['ensemble']['accuracy'], min_accuracy)
    
    def test_regression_pipeline(self):
        """Test complete regression pipeline"""
        # Create data
        np.random.seed(TEST_RANDOM_SEED)
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            noise=5,
            random_state=TEST_RANDOM_SEED
        )
        
        # Scale to win range
        y = 81 + (y - y.mean()) / y.std() * 10
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=TEST_RANDOM_SEED
        )
        
        # Train and evaluate model
        model = BaseballRegressor(model_type='lightgbm')
        model.train(X_train, y_train)
        
        # Get predictions with bounds
        pred_dict = model.predict_with_bounds(X_test)
        predictions = pred_dict['predictions']
        lower = pred_dict['lower_bound']
        upper = pred_dict['upper_bound']
        
        # Evaluate
        metrics = RegressionMetrics(y_test, predictions)
        results = metrics.metrics
        
        # Check reasonable performance
        self.assertLess(results['rmse'], 15)
        self.assertGreater(results['r2'], 0.5)
        
        # Check bounds coverage
        coverage = ((y_test >= lower) & (y_test <= upper)).mean()
        self.assertGreater(coverage, 0.8)


if __name__ == '__main__':
    unittest.main()