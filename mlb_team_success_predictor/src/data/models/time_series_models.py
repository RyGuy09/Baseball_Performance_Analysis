"""
Time series models for forecasting team performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Prophet import with fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")


class TeamPerformanceForecaster(BaseModel):
    """Time series forecasting for team performance metrics"""
    
    def __init__(self, model_type: str = 'arima',
                 forecast_metric: str = 'wins',
                 season_length: int = 162):
        """
        Initialize forecaster
        
        Args:
            model_type: 'arima', 'prophet', 'exponential_smoothing', or 'ensemble'
            forecast_metric: Metric to forecast ('wins', 'run_differential', etc.)
            season_length: Number of games in a season
        """
        super().__init__(model_name=f"Forecaster_{model_type}_{forecast_metric}")
        
        self.model_type = model_type
        self.forecast_metric = forecast_metric
        self.season_length = season_length
        self.models = {}  # Store models per team
        
    def prepare_time_series_data(self, df: pd.DataFrame, 
                                team_name: str) -> pd.DataFrame:
        """
        Prepare data for time series analysis
        
        Args:
            df: Historical data
            team_name: Team to analyze
            
        Returns:
            Time series DataFrame
        """
        # Filter for specific team
        team_df = df[df['team_name'] == team_name].copy()
        
        # Sort by year
        team_df = team_df.sort_values('year')
        
        # Create datetime index (using year as period)
        team_df['date'] = pd.to_datetime(team_df['year'], format='%Y')
        team_df = team_df.set_index('date')
        
        # Select metric
        if self.forecast_metric not in team_df.columns:
            raise ValueError(f"Metric {self.forecast_metric} not found in data")
        
        return team_df[[self.forecast_metric]]
    
    def train(self, df: pd.DataFrame, team_names: Optional[List[str]] = None):
        """
        Train forecasting models
        
        Args:
            df: Historical data
            team_names: Teams to train models for (None = all teams)
        """
        if team_names is None:
            team_names = df['team_name'].unique()
        
        for team in team_names:
            logger.info(f"Training {self.model_type} model for {team}")
            
            try:
                # Prepare time series
                ts_data = self.prepare_time_series_data(df, team)
                
                if len(ts_data) < 10:
                    logger.warning(f"Insufficient data for {team}. Skipping.")
                    continue
                
                # Train model based on type
                if self.model_type == 'arima':
                    model = self._train_arima(ts_data)
                elif self.model_type == 'prophet' and PROPHET_AVAILABLE:
                    model = self._train_prophet(ts_data)
                elif self.model_type == 'exponential_smoothing':
                    model = self._train_exp_smoothing(ts_data)
                elif self.model_type == 'ensemble':
                    model = self._train_ensemble(ts_data)
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
                
                self.models[team] = model
                
            except Exception as e:
                logger.error(f"Error training model for {team}: {str(e)}")
    
    def _train_arima(self, ts_data: pd.DataFrame) -> ARIMA:
        """Train ARIMA model with auto-selection of parameters"""
        # Simple parameter selection (in practice, use auto_arima)
        best_aic = np.inf
        best_params = (1, 0, 1)
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    if p == 0 and q == 0:
                        continue
                    
                    try:
                        model = ARIMA(ts_data[self.forecast_metric], 
                                     order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        # Fit final model
        final_model = ARIMA(ts_data[self.forecast_metric], order=best_params)
        fitted_model = final_model.fit()
        
        logger.info(f"ARIMA{best_params} selected with AIC: {best_aic:.2f}")
        
        return fitted_model
    
    def _train_prophet(self, ts_data: pd.DataFrame) -> Prophet:
        """Train Prophet model"""
        # Prepare data for Prophet
        prophet_df = ts_data.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize Prophet with custom parameters for sports data
        model = Prophet(
            yearly_seasonality=False,  # No within-year seasonality
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,  # More conservative trend changes
            interval_width=0.80
        )
        
        # Fit model
        model.fit(prophet_df)
        
        return model
    
    def _train_exp_smoothing(self, ts_data: pd.DataFrame) -> ExponentialSmoothing:
        """Train Exponential Smoothing model"""
        # Simple exponential smoothing for annual data
        model = ExponentialSmoothing(
            ts_data[self.forecast_metric],
            trend='add',
            seasonal=None,  # No seasonality for annual data
            initialization_method='estimated'
        )
        
        fitted_model = model.fit()
        
        return fitted_model
    
    def _train_ensemble(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble of time series models"""
        ensemble = {}
        
        # Train multiple models
        try:
            ensemble['arima'] = self._train_arima(ts_data)
        except:
            logger.warning("ARIMA failed in ensemble")
        
        try:
            ensemble['exp_smoothing'] = self._train_exp_smoothing(ts_data)
        except:
            logger.warning("Exponential smoothing failed in ensemble")
        
        if PROPHET_AVAILABLE:
            try:
                ensemble['prophet'] = self._train_prophet(ts_data)
            except:
                logger.warning("Prophet failed in ensemble")
        
        return ensemble
    
    def predict(self, team_name: str, periods: int = 3) -> pd.DataFrame:
        """
        Forecast future performance
        
        Args:
            team_name: Team to forecast
            periods: Number of periods (years) to forecast
            
        Returns:
            DataFrame with forecasts
        """
        if team_name not in self.models:
            raise ValueError(f"No model trained for {team_name}")
        
        model = self.models[team_name]
        
        if self.model_type == 'arima':
            return self._predict_arima(model, periods)
        elif self.model_type == 'prophet' and PROPHET_AVAILABLE:
            return self._predict_prophet(model, periods)
        elif self.model_type == 'exponential_smoothing':
            return self._predict_exp_smoothing(model, periods)
        elif self.model_type == 'ensemble':
            return self._predict_ensemble(model, periods)
    
    def _predict_arima(self, model: ARIMA, periods: int) -> pd.DataFrame:
        """Generate ARIMA forecasts"""
        forecast_result = model.forecast(steps=periods)
        
        # Get prediction intervals
        forecast_df = pd.DataFrame({
            'forecast': forecast_result,
            'year': range(model.nobs + 2024 - len(model.model.endog) + 1, 
                         model.nobs + 2024 - len(model.model.endog) + periods + 1)
        })
        
        return forecast_df
    
    def _predict_prophet(self, model: Prophet, periods: int) -> pd.DataFrame:
        """Generate Prophet forecasts"""
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='Y')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract relevant columns
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_df.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
        forecast_df['year'] = forecast_df['date'].dt.year
        
        return forecast_df
    
    def _predict_exp_smoothing(self, model: ExponentialSmoothing, 
                              periods: int) -> pd.DataFrame:
        """Generate Exponential Smoothing forecasts"""
        forecast = model.forecast(steps=periods)
        
        forecast_df = pd.DataFrame({
            'forecast': forecast,
            'year': range(2025, 2025 + periods)
        })
        
        return forecast_df
    
    def _predict_ensemble(self, models: Dict[str, Any], periods: int) -> pd.DataFrame:
        """Generate ensemble forecasts"""
        forecasts = []
        
        for name, model in models.items():
            try:
                if name == 'arima':
                    pred = self._predict_arima(model, periods)['forecast'].values
                elif name == 'exp_smoothing':
                    pred = self._predict_exp_smoothing(model, periods)['forecast'].values
                elif name == 'prophet' and PROPHET_AVAILABLE:
                    pred = self._predict_prophet(model, periods)['forecast'].values
                
                forecasts.append(pred)
            except:
                logger.warning(f"Failed to get prediction from {name}")
        
        # Average forecasts
        if forecasts:
            ensemble_forecast = np.mean(forecasts, axis=0)
            
            forecast_df = pd.DataFrame({
                'forecast': ensemble_forecast,
                'year': range(2025, 2025 + periods),
                'std': np.std(forecasts, axis=0) if len(forecasts) > 1 else np.zeros(periods)
            })
            
            return forecast_df
        else:
            raise ValueError("No successful forecasts in ensemble")


class SeasonalDecomposer:
    """Decompose team performance into trend and random components"""
    
    def __init__(self, method: str = 'additive'):
        """
        Initialize decomposer
        
        Args:
            method: 'additive' or 'multiplicative'
        """
        self.method = method
        self.decomposition_results = {}
    
    def decompose(self, df: pd.DataFrame, team_name: str, 
                 metric: str = 'wins') -> Dict[str, pd.Series]:
        """
        Decompose team performance
        
        Args:
            df: Historical data
            team_name: Team to analyze
            metric: Metric to decompose
            
        Returns:
            Dictionary with trend and residual components
        """
        # Prepare time series
        team_df = df[df['team_name'] == team_name].sort_values('year')
        ts = team_df.set_index('year')[metric]
        
        # Simple decomposition for annual data (no seasonal component)
        # Calculate trend using moving average
        window = min(5, len(ts) // 2)
        trend = ts.rolling(window=window, center=True).mean()
        
        # Calculate residuals
        if self.method == 'additive':
            residuals = ts - trend
        else:  # multiplicative
            residuals = ts / trend
        
        # Store results
        self.decomposition_results[team_name] = {
            'observed': ts,
            'trend': trend,
            'residuals': residuals
        }
        
        return self.decomposition_results[team_name]
    
    def plot_decomposition(self, team_name: str):
        """Plot decomposition results"""
        if team_name not in self.decomposition_results:
            raise ValueError(f"No decomposition results for {team_name}")
        
        import matplotlib.pyplot as plt
        
        results = self.decomposition_results[team_name]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Observed
        results['observed'].plot(ax=axes[0], title=f'{team_name} - Observed')
        axes[0].set_ylabel('Wins')
        
        # Trend
        results['trend'].plot(ax=axes[1], title='Trend', color='green')
        axes[1].set_ylabel('Wins')
        
        # Residuals
        results['residuals'].plot(ax=axes[2], title='Residuals', color='red')
        axes[2].set_ylabel('Residual')
        
        plt.tight_layout()
        return fig


class PerformanceTrendAnalyzer:
    """Analyze long-term performance trends"""
    
    def __init__(self):
        self.trend_results = {}
    
    def analyze_trends(self, df: pd.DataFrame, 
                      metrics: List[str] = ['wins', 'run_differential']) -> pd.DataFrame:
        """
        Analyze performance trends across all teams
        
        Args:
            df: Historical data
            metrics: Metrics to analyze
            
        Returns:
            DataFrame with trend analysis results
        """
        results = []
        
        for team in df['team_name'].unique():
            team_df = df[df['team_name'] == team].sort_values('year')
            
            if len(team_df) < 5:
                continue
            
            team_results = {'team_name': team}
            
            for metric in metrics:
                if metric not in team_df.columns:
                    continue
                
                # Calculate trend statistics
                values = team_df[metric].values
                years = np.arange(len(values))
                
                # Linear trend
                slope, intercept = np.polyfit(years, values, 1)
                
                # Recent performance (last 5 years)
                recent_avg = values[-5:].mean() if len(values) >= 5 else values.mean()
                historical_avg = values.mean()
                
                # Volatility
                volatility = values.std()
                
                team_results.update({
                    f'{metric}_trend': slope,
                    f'{metric}_recent_avg': recent_avg,
                    f'{metric}_historical_avg': historical_avg,
                    f'{metric}_improvement': recent_avg - historical_avg,
                    f'{metric}_volatility': volatility
                })
            
            results.append(team_results)
        
        return pd.DataFrame(results)
    
    def identify_trending_teams(self, df: pd.DataFrame, 
                               metric: str = 'wins',
                               trend_threshold: float = 1.0) -> Dict[str, List[str]]:
        """
        Identify teams with strong positive or negative trends
        
        Args:
            df: Historical data
            metric: Metric to analyze
            trend_threshold: Minimum trend slope to be considered significant
            
        Returns:
            Dictionary with 'improving' and 'declining' team lists
        """
        trend_df = self.analyze_trends(df, [metric])
        
        improving = trend_df[
            trend_df[f'{metric}_trend'] >= trend_threshold
        ]['team_name'].tolist()
        
        declining = trend_df[
            trend_df[f'{metric}_trend'] <= -trend_threshold
        ]['team_name'].tolist()
        
        return {
            'improving': improving,
            'declining': declining,
            'stable': [t for t in trend_df['team_name'] 
                      if t not in improving and t not in declining]
        }