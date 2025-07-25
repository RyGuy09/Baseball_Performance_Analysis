"""
Interactive visualizations using Plotly
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
from datetime import datetime

from ..utils.constants import MILESTONE_THRESHOLDS, ERA_DEFINITIONS
from ..utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

# Set default template
pio.templates.default = "plotly_white"


class InteractiveDashboard:
    """Create interactive dashboards for MLB data"""
    
    def __init__(self, theme: str = 'plotly_white',
                 save_dir: Optional[Path] = None):
        """
        Initialize dashboard creator
        
        Args:
            theme: Plotly theme
            save_dir: Directory to save dashboards
        """
        self.theme = theme
        self.save_dir = Path(save_dir) if save_dir else Path('dashboards')
        ensure_dir(self.save_dir)
        
        pio.templates.default = theme
    
    def create_team_performance_dashboard(self, df: pd.DataFrame,
                                        team_name: str,
                                        save: bool = True) -> go.Figure:
        """
        Create interactive dashboard for team performance
        
        Args:
            df: Data with team statistics
            team_name: Team to analyze
            save: Whether to save the dashboard
            
        Returns:
            Plotly figure
        """
        team_df = df[df['team_name'] == team_name].sort_values('year')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Wins/Losses Over Time', 'Run Differential',
                          'Win Percentage Trend', 'Runs Scored vs Allowed'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Wins/Losses over time
        fig.add_trace(
            go.Scatter(x=team_df['year'], y=team_df['wins'],
                      mode='lines+markers', name='Wins',
                      line=dict(color='green', width=2)),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=team_df['year'], y=team_df['losses'],
                      mode='lines+markers', name='Losses',
                      line=dict(color='red', width=2)),
            row=1, col=1, secondary_y=False
        )
        
        # Add .500 line
        fig.add_hline(y=81, line_dash="dash", line_color="gray",
                     annotation_text=".500", row=1, col=1)
        
        # 2. Run differential bar chart
        colors = ['red' if x < 0 else 'green' for x in team_df['run_differential']]
        fig.add_trace(
            go.Bar(x=team_df['year'], y=team_df['run_differential'],
                   marker_color=colors, name='Run Diff',
                   hovertemplate='Year: %{x}<br>Run Diff: %{y}<extra></extra>'),
            row=1, col=2
        )
        
        # 3. Win percentage with trend
        fig.add_trace(
            go.Scatter(x=team_df['year'], y=team_df['winning_percentage'],
                      mode='markers', name='Win %',
                      marker=dict(size=8, color='blue')),
            row=2, col=1
        )
        
        # Add moving average
        if len(team_df) >= 5:
            ma = team_df['winning_percentage'].rolling(window=5, center=True).mean()
            fig.add_trace(
                go.Scatter(x=team_df['year'], y=ma,
                          mode='lines', name='5-yr MA',
                          line=dict(color='red', width=2)),
                row=2, col=1
            )
        
        # 4. Runs scored vs allowed
        fig.add_trace(
            go.Scatter(x=team_df['runs_allowed'], y=team_df['runs_scored'],
                      mode='markers', name='Season',
                      marker=dict(size=10, 
                                color=team_df['year'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Year")),
                      text=[f"{year}" for year in team_df['year']],
                      hovertemplate='Year: %{text}<br>RS: %{y}<br>RA: %{x}<extra></extra>'),
            row=2, col=2
        )
        
        # Add reference line (equal runs)
        min_runs = min(team_df['runs_scored'].min(), team_df['runs_allowed'].min())
        max_runs = max(team_df['runs_scored'].max(), team_df['runs_allowed'].max())
        fig.add_trace(
            go.Scatter(x=[min_runs, max_runs], y=[min_runs, max_runs],
                      mode='lines', name='Equal',
                      line=dict(color='gray', dash='dash')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"{team_name} Performance Dashboard",
            showlegend=True,
            height=800,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_xaxes(title_text="Runs Allowed", row=2, col=2)
        
        fig.update_yaxes(title_text="Games", row=1, col=1)
        fig.update_yaxes(title_text="Run Differential", row=1, col=2)
        fig.update_yaxes(title_text="Win Percentage", row=2, col=1)
        fig.update_yaxes(title_text="Runs Scored", row=2, col=2)
        
        if save:
            save_path = self.save_dir / f'{team_name.replace(" ", "_")}_dashboard.html'
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def create_season_comparison_dashboard(self, df: pd.DataFrame,
                                         years: List[int],
                                         save: bool = True) -> go.Figure:
        """
        Create dashboard comparing multiple seasons
        
        Args:
            df: Data with season statistics
            years: Years to compare
            save: Whether to save the dashboard
            
        Returns:
            Plotly figure
        """
        # Filter data
        season_df = df[df['year'].isin(years)]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Team Performance Distribution', 'Run Production',
                          'League Standings', 'Performance Metrics'),
            specs=[[{"type": "box"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Win distribution by year
        for year in years:
            year_data = season_df[season_df['year'] == year]
            fig.add_trace(
                go.Box(y=year_data['wins'], name=str(year),
                      boxpoints='all', jitter=0.3),
                row=1, col=1
            )
        
        # 2. Run production scatter
        fig.add_trace(
            go.Scatter(x=season_df['runs_scored'], y=season_df['runs_allowed'],
                      mode='markers',
                      marker=dict(size=10,
                                color=season_df['year'].astype(str),
                                symbol=season_df['year'].astype(str)),
                      text=season_df['team_name'],
                      name='Teams',
                      hovertemplate='%{text}<br>RS: %{x}<br>RA: %{y}<extra></extra>'),
            row=1, col=2
        )
        
        # 3. Top teams by year
        for i, year in enumerate(years):
            year_data = season_df[season_df['year'] == year].nlargest(5, 'wins')
            fig.add_trace(
                go.Bar(x=year_data['team_name'], y=year_data['wins'],
                      name=str(year), 
                      marker_color=px.colors.qualitative.Set1[i % 10]),
                row=2, col=1
            )
        
        # 4. Performance metrics scatter matrix
        metrics_df = season_df[['winning_percentage', 'run_differential', 'year']]
        for year in years:
            year_data = metrics_df[metrics_df['year'] == year]
            fig.add_trace(
                go.Scatter(x=year_data['run_differential'], 
                          y=year_data['winning_percentage'],
                          mode='markers', name=str(year),
                          marker=dict(size=8)),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Season Comparison: {', '.join(map(str, years))}",
            showlegend=True,
            height=800,
            barmode='group'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_xaxes(title_text="Runs Scored", row=1, col=2)
        fig.update_xaxes(title_text="Team", row=2, col=1)
        fig.update_xaxes(title_text="Run Differential", row=2, col=2)
        
        fig.update_yaxes(title_text="Wins", row=1, col=1)
        fig.update_yaxes(title_text="Runs Allowed", row=1, col=2)
        fig.update_yaxes(title_text="Wins", row=2, col=1)
        fig.update_yaxes(title_text="Win Percentage", row=2, col=2)
        
        if save:
            save_path = self.save_dir / f'season_comparison_{"_".join(map(str, years))}.html'
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def create_prediction_results_dashboard(self, actual: pd.DataFrame,
                                          predictions: pd.DataFrame,
                                          model_name: str = 'Model',
                                          save: bool = True) -> go.Figure:
        """
        Create interactive dashboard for prediction results
        
        Args:
            actual: Actual values
            predictions: Predicted values with team names
            model_name: Name of the model
            save: Whether to save the dashboard
            
        Returns:
            Plotly figure
        """
        # Merge actual and predictions
        results = actual.merge(predictions, on='team_name', suffixes=('_actual', '_pred'))
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Predicted vs Actual Wins', 'Prediction Errors',
                          'Team Rankings Comparison', 'Error Distribution'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "histogram"}]]
        )
        
        # 1. Predicted vs Actual scatter
        fig.add_trace(
            go.Scatter(x=results['wins_actual'], y=results['wins_pred'],
                      mode='markers', name='Teams',
                      marker=dict(size=10, 
                                color=results['wins_pred'] - results['wins_actual'],
                                colorscale='RdBu', 
                                showscale=True,
                                colorbar=dict(title="Error")),
                      text=results['team_name'],
                      hovertemplate='%{text}<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'),
            row=1, col=1
        )
        
        # Add perfect prediction line
        min_wins = min(results['wins_actual'].min(), results['wins_pred'].min())
        max_wins = max(results['wins_actual'].max(), results['wins_pred'].max())
        fig.add_trace(
            go.Scatter(x=[min_wins, max_wins], y=[min_wins, max_wins],
                      mode='lines', name='Perfect',
                      line=dict(color='black', dash='dash')),
            row=1, col=1
        )
        
        # 2. Prediction errors by team
        errors = results['wins_pred'] - results['wins_actual']
        sorted_results = results.iloc[errors.abs().argsort()[::-1]].head(10)
        
        fig.add_trace(
            go.Bar(x=sorted_results['team_name'],
                   y=sorted_results['wins_pred'] - sorted_results['wins_actual'],
                   marker_color=['red' if x < 0 else 'green' 
                               for x in sorted_results['wins_pred'] - sorted_results['wins_actual']],
                   name='Error'),
            row=1, col=2
        )
        
        # 3. Rankings table
        results['rank_actual'] = results['wins_actual'].rank(ascending=False)
        results['rank_pred'] = results['wins_pred'].rank(ascending=False)
        results['rank_diff'] = results['rank_pred'] - results['rank_actual']
        
        top_teams = results.nsmallest(10, 'rank_actual')[
            ['team_name', 'wins_actual', 'wins_pred', 'rank_actual', 'rank_pred', 'rank_diff']
        ].round(1)
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Team', 'Actual Wins', 'Pred Wins', 
                                  'Actual Rank', 'Pred Rank', 'Rank Diff'],
                           align='left'),
                cells=dict(values=[top_teams[col] for col in top_teams.columns],
                          align='left')),
            row=2, col=1
        )
        
        # 4. Error distribution
        fig.add_trace(
            go.Histogram(x=results['wins_pred'] - results['wins_actual'],
                        nbinsx=20, name='Errors'),
            row=2, col=2
        )
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(results['wins_actual'], results['wins_pred']))
        mae = mean_absolute_error(results['wins_actual'], results['wins_pred'])
        r2 = r2_score(results['wins_actual'], results['wins_pred'])
        
        # Update layout
        fig.update_layout(
            title=f"{model_name} Prediction Results<br>" +
                  f"<sub>RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}</sub>",
            showlegend=False,
            height=800
        )
        
        # Update axes
        fig.update_xaxes(title_text="Actual Wins", row=1, col=1)
        fig.update_xaxes(title_text="Team", row=1, col=2)
        fig.update_xaxes(title_text="Prediction Error", row=2, col=2)
        
        fig.update_yaxes(title_text="Predicted Wins", row=1, col=1)
        fig.update_yaxes(title_text="Prediction Error", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        if save:
            save_path = self.save_dir / f'{model_name.lower()}_prediction_results.html'
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def create_feature_exploration_dashboard(self, df: pd.DataFrame,
                                           features: List[str],
                                           target: str = 'wins',
                                           save: bool = True) -> go.Figure:
        """
        Create interactive feature exploration dashboard
        
        Args:
            df: DataFrame with features
            features: List of features to explore
            target: Target variable
            save: Whether to save the dashboard
            
        Returns:
            Plotly figure
        """
        # Create correlation matrix
        corr_features = features + [target]
        corr_matrix = df[corr_features].corr()
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Correlation Matrix', 'Feature vs Target',
                          'Feature Distributions', 'Parallel Coordinates'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # 1. Correlation heatmap
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values,
                      x=corr_matrix.columns,
                      y=corr_matrix.columns,
                      colorscale='RdBu',
                      zmid=0,
                      text=corr_matrix.values.round(2),
                      texttemplate='%{text}',
                      textfont={"size": 10}),
            row=1, col=1
        )
        
        # 2. Feature vs Target (for first feature)
        if features:
            fig.add_trace(
                go.Scatter(x=df[features[0]], y=df[target],
                          mode='markers',
                          marker=dict(size=5, opacity=0.6),
                          name=features[0]),
                row=1, col=2
            )
        
        # 3. Feature distributions
        for feature in features[:3]:  # Limit to 3 features
            fig.add_trace(
                go.Box(y=df[feature], name=feature),
                row=2, col=1
            )
        
        # 4. Parallel coordinates
        # Normalize features for parallel coordinates
        norm_df = df[features + [target]].copy()
        for col in norm_df.columns:
            norm_df[col] = (norm_df[col] - norm_df[col].min()) / (norm_df[col].max() - norm_df[col].min())
        
        # Create dimensions for parallel coordinates
        dimensions = []
        for col in features[:5] + [target]:  # Limit to 5 features + target
            dimensions.append(
                dict(label=col,
                     values=norm_df[col],
                     range=[0, 1])
            )
        
        # Add parallel coordinates as a separate trace
        parcoords = go.Parcoords(
            line=dict(color=norm_df[target],
                     colorscale='Viridis',
                     showscale=True,
                     colorbar=dict(title=target)),
            dimensions=dimensions
        )
        
        # Create a new figure for parallel coordinates
        fig_parcoords = go.Figure(data=[parcoords])
        fig_parcoords.update_layout(title="Feature Relationships - Parallel Coordinates")
        
        # Update main figure layout
        fig.update_layout(
            title="Feature Exploration Dashboard",
            showlegend=True,
            height=800
        )
        
        # Update axes
        fig.update_xaxes(title_text=features[0] if features else "Feature", row=1, col=2)
        fig.update_yaxes(title_text=target, row=1, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        if save:
            save_path = self.save_dir / 'feature_exploration_dashboard.html'
            fig.write_html(save_path)
            
            # Save parallel coordinates separately
            save_path_parcoords = self.save_dir / 'feature_parallel_coordinates.html'
            fig_parcoords.write_html(save_path_parcoords)
            
            logger.info(f"Dashboards saved to {self.save_dir}")
        
        return fig
    
    def create_time_series_dashboard(self, df: pd.DataFrame,
                                   metric: str = 'wins',
                                   teams: Optional[List[str]] = None,
                                   save: bool = True) -> go.Figure:
        """
        Create time series analysis dashboard
        
        Args:
            df: DataFrame with time series data
            metric: Metric to analyze
            teams: Specific teams to include
            save: Whether to save the dashboard
            
        Returns:
            Plotly figure
        """
        # Filter teams if specified
        if teams:
            df_filtered = df[df['team_name'].isin(teams)]
        else:
            # Select top teams by average metric
            top_teams = df.groupby('team_name')[metric].mean().nlargest(5).index
            df_filtered = df[df['team_name'].isin(top_teams)]
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each team
        for team in df_filtered['team_name'].unique():
            team_data = df_filtered[df_filtered['team_name'] == team].sort_values('year')
            
            # Add main line
            fig.add_trace(
                go.Scatter(x=team_data['year'], y=team_data[metric],
                          mode='lines+markers',
                          name=team,
                          line=dict(width=2),
                          marker=dict(size=6),
                          hovertemplate=f'{team}<br>Year: %{{x}}<br>{metric}: %{{y}}<extra></extra>')
            )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(count=20, label="20y", step="year", stepmode="backward"),
                    dict(count=30, label="30y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"{metric.title()} Over Time - Interactive Timeline",
            xaxis_title="Year",
            yaxis_title=metric.title(),
            hovermode='x unified',
            height=600
        )
        
        # Add era annotations
        for era_name, era_info in ERA_DEFINITIONS.items():
            if era_name != 'unknown':
                fig.add_vrect(
                    x0=era_info['years'][0], x1=era_info['years'][1],
                    fillcolor="gray", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text=era_name.replace('_', ' ').title(),
                    annotation_position="top left"
                )
        
        if save:
            save_path = self.save_dir / f'time_series_{metric}_dashboard.html'
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig


# Convenience functions
def create_team_dashboard(df: pd.DataFrame, team_name: str,
                         save_dir: Optional[Path] = None) -> go.Figure:
    """Quick function to create team dashboard"""
    dashboard = InteractiveDashboard(save_dir=save_dir)
    return dashboard.create_team_performance_dashboard(df, team_name)


def create_season_dashboard(df: pd.DataFrame, years: List[int],
                           save_dir: Optional[Path] = None) -> go.Figure:
    """Quick function to create season comparison dashboard"""
    dashboard = InteractiveDashboard(save_dir=save_dir)
    return dashboard.create_season_comparison_dashboard(df, years)


def create_prediction_dashboard(actual: pd.DataFrame, predictions: pd.DataFrame,
                              model_name: str = 'Model',
                              save_dir: Optional[Path] = None) -> go.Figure:
    """Quick function to create prediction results dashboard"""
    dashboard = InteractiveDashboard(save_dir=save_dir)
    return dashboard.create_prediction_results_dashboard(actual, predictions, model_name)


def create_comparison_dashboard(df: pd.DataFrame,
                              features: List[str],
                              target: str = 'wins',
                              save_dir: Optional[Path] = None) -> go.Figure:
    """Quick function to create feature comparison dashboard"""
    dashboard = InteractiveDashboard(save_dir=save_dir)
    return dashboard.create_feature_exploration_dashboard(df, features, target)


if __name__ == "__main__":
    # Example usage
    # Create sample data
    sample_data = pd.DataFrame({
        'team_name': ['Yankees', 'Red Sox', 'Dodgers'] * 10,
        'year': list(range(2014, 2024)) * 3,
        'wins': np.random.randint(70, 100, 30),
        'losses': np.random.randint(62, 92, 30),
        'runs_scored': np.random.randint(650, 900, 30),
        'runs_allowed': np.random.randint(600, 850, 30)
    })
    
    sample_data['winning_percentage'] = sample_data['wins'] / (sample_data['wins'] + sample_data['losses'])
    sample_data['run_differential'] = sample_data['runs_scored'] - sample_data['runs_allowed']
    
    # Create dashboards
    dashboard = InteractiveDashboard()
    
    # Team dashboard
    fig1 = dashboard.create_team_performance_dashboard(sample_data, 'Yankees', save=False)
    fig1.show()
    
    # Season comparison
    fig2 = dashboard.create_season_comparison_dashboard(sample_data, [2020, 2021, 2022], save=False)
    fig2.show()