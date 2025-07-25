# MLB Team Success Predictor - Model Information

## Model Files Overview

This directory contains the trained machine learning models and data scalers used by the MLB Team Success Predictor application.

### Saved Models

#### 1. `division_classifier.pkl`
- **Type**: XGBoost Classifier
- **Purpose**: Predicts whether a team will win their division
- **Features**: 20 engineered features including performance metrics, historical data, and era-adjusted statistics
- **Performance**: 
  - Accuracy: ~85%
  - ROC-AUC: ~0.92
  - Precision/Recall balanced for practical use

#### 2. `wins_regressor.pkl`
- **Type**: LightGBM Regressor
- **Purpose**: Predicts total season wins for each team
- **Features**: 22 engineered features including previous season data, moving averages, and advanced metrics
- **Performance**:
  - RMSE: ~6.2 wins
  - R²: ~0.86
  - 68% predictions within 5 wins
  - 91% predictions within 10 wins

#### 3. `milestone_predictor.pkl`
- **Type**: Multi-Output Random Forest Classifier
- **Purpose**: Predicts probability of achieving various milestones
- **Milestones**:
  - 90+ wins
  - 100+ wins
  - Playoff appearance
  - 800+ runs scored
  - Team ERA under 4.00
- **Performance**: 
  - Overall accuracy: ~86%
  - Well-calibrated probabilities

### Scalers

#### 1. `classification_scaler.pkl`
- **Type**: StandardScaler
- **Purpose**: Normalizes features for division classification
- **Fitted on**: 1,500+ team seasons

#### 2. `regression_scaler.pkl`
- **Type**: StandardScaler
- **Purpose**: Normalizes features for win prediction
- **Fitted on**: 1,500+ team seasons

#### 3. `milestone_scaler.pkl`
- **Type**: StandardScaler
- **Purpose**: Normalizes features for milestone prediction
- **Fitted on**: 1,500+ team seasons

## Model Training Details

### Data
- Historical MLB data from 1901-2024
- Modern era (2006+) emphasized for recent predictions
- Features engineered using domain expertise

### Validation Strategy
- Time-based train/validation/test split
- Cross-validation for hyperparameter tuning
- Out-of-time testing on recent seasons

### Update Schedule
- Models retrained annually after season completion
- Mid-season updates possible for significant changes
- Version control maintained for all models

## Usage

```python
import joblib

# Load a model
model_dict = joblib.load('saved_models/division_classifier.pkl')
model = model_dict['model']
feature_names = model_dict['feature_names']

# Load corresponding scaler
scaler_dict = joblib.load('scalers/classification_scaler.pkl')
scaler = scaler_dict['scaler']

# Make predictions
X_scaled = scaler.transform(X[feature_names])
predictions = model.predict(X_scaled)