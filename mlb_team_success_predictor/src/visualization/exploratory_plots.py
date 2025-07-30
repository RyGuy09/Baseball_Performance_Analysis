"""
Exploratory data analysis visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..data.utils.constants import ERA_DEFINITIONS, MILESTONE_THRESHOLDS
from ..data.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ExploratoryPlotter:
    """Create exploratory data analysis plots"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6),
                 style: str = 'darkgrid',
                 save_dir: Optional[Path] = None):
        """
        Initialize plotter
        
        Args:
            figsize: Default figure size
            style: Seaborn style
            save_dir: Directory to save plots
        """
        self.figsize = figsize
        self.style = style
        self.save_dir = Path(save_dir) if save_dir else Path('plots/eda')
        ensure_dir(self.save_dir)
        
        sns.set_style(style)
    
    def plot_team_performance_overview(self, df: pd.DataFrame,
                                     team_name: str,
                                     save: bool = True) -> plt.Figure:
        """
        Plot comprehensive team performance over time
        
        Args:
            df: Data with team statistics
            team_name: Team to plot
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        team_df = df[df['team_name'] == team_name].sort_values('year')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{team_name} Performance Overview', fontsize=16)
        
        # 1. Wins/Losses over time
        ax = axes[0, 0]
        ax.plot(team_df['year'], team_df['wins'], 'b-', label='Wins', linewidth=2)
        ax.plot(team_df['year'], team_df['losses'], 'r-', label='Losses', linewidth=2)
        ax.axhline(y=81, color='gray', linestyle='--', alpha=0.5, label='.500 record')
        ax.set_xlabel('Year')
        ax.set_ylabel('Games')
        ax.set_title('Wins and Losses Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Run differential
        ax = axes[0, 1]
        colors = ['red' if x < 0 else 'green' for x in team_df['run_differential']]
        ax.bar(team_df['year'], team_df['run_differential'], color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Year')
        ax.set_ylabel('Run Differential')
        ax.set_title('Run Differential by Season')
        ax.grid(True, alpha=0.3)
        
        # 3. Winning percentage with moving average
        ax = axes[1, 0]
        ax.plot(team_df['year'], team_df['winning_percentage'], 'o-', 
                label='Win %', alpha=0.7)
        
        # Add 5-year moving average
        if len(team_df) >= 5:
            ma = team_df['winning_percentage'].rolling(window=5, center=True).mean()
            ax.plot(team_df['year'], ma, 'r-', linewidth=2, 
                    label='5-year MA')
        
        ax.axhline(y=0.500, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Year')
        ax.set_ylabel('Winning Percentage')
        ax.set_title('Winning Percentage Trend')
        ax.set_ylim(0.3, 0.7)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Playoff appearances (approximation)
        ax = axes[1, 1]
        playoff_threshold = team_df.groupby('year')['wins'].transform(
            lambda x: x.quantile(0.75)
        )
        playoffs = (team_df['wins'] >= playoff_threshold).astype(int)
        
        ax.bar(team_df['year'], playoffs, color='gold', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Playoff Appearance')
        ax.set_title('Estimated Playoff Appearances')
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{team_name.replace(" ", "_")}_overview.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_era_comparison(self, df: pd.DataFrame,
                           metric: str = 'winning_percentage',
                           save: bool = True) -> plt.Figure:
        """
        Compare team performance across different eras
        
        Args:
            df: Data with era information
            metric: Metric to compare
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        # Add era if not present
        if 'era' not in df.columns:
            df['era'] = df['year'].apply(self._get_era)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Box plot by era
        era_order = ['dead_ball', 'live_ball', 'integration', 'expansion', 
                     'free_agency', 'steroid', 'modern']
        existing_eras = [era for era in era_order if era in df['era'].unique()]
        
        sns.boxplot(data=df, x='era', y=metric, order=existing_eras, ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_title(f'{metric.replace("_", " ").title()} by Era')
        ax1.grid(True, alpha=0.3)
        
        # 2. Violin plot with points
        sns.violinplot(data=df, x='era', y=metric, order=existing_eras, 
                       inner='quartile', ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_title(f'{metric.replace("_", " ").title()} Distribution by Era')
        ax2.grid(True, alpha=0.3)
        
        # Add era descriptions
        era_info = []
        for era in existing_eras:
            if era in ERA_DEFINITIONS:
                years = ERA_DEFINITIONS[era]['years']
                era_info.append(f"{era}: {years[0]}-{years[1]}")
        
        fig.text(0.5, -0.05, ' | '.join(era_info), ha='center', fontsize=10)
        
        plt.suptitle(f'Era Comparison: {metric.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'era_comparison_{metric}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_relationships(self, df: pd.DataFrame,
                                 features: List[str],
                                 target: str = 'wins',
                                 save: bool = True) -> plt.Figure:
        """
        Plot relationships between features and target
        
        Args:
            df: DataFrame with features
            features: List of features to plot
            target: Target variable
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Scatter plot with regression line
            sns.regplot(data=df, x=feature, y=target, ax=ax, 
                       scatter_kws={'alpha': 0.5})
            
            # Calculate correlation
            corr = df[feature].corr(df[target])
            ax.set_title(f'{feature} vs {target}\n(r = {corr:.3f})')
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel(target.replace('_', ' ').title())
            
            # Add reference lines for milestone values if applicable
            if target == 'wins':
                for milestone, value in MILESTONE_THRESHOLDS['wins'].items():
                    if milestone in ['good', 'excellent']:
                        ax.axhline(y=value, color='red', linestyle='--', 
                                  alpha=0.3, label=milestone)
        
        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Feature Relationships', fontsize=14)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'feature_relationships_{target}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distribution_analysis(self, df: pd.DataFrame,
                                 columns: List[str],
                                 save: bool = True) -> plt.Figure:
        """
        Plot distribution analysis for multiple columns
        
        Args:
            df: DataFrame
            columns: Columns to analyze
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 3, figsize=(15, 4 * n_cols))
        
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(columns):
            data = df[col].dropna()
            
            # 1. Histogram with KDE
            ax = axes[i, 0]
            ax.hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')
            data.plot.kde(ax=ax, color='red', linewidth=2)
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel('Density')
            
            # 2. Box plot
            ax = axes[i, 1]
            box_data = pd.DataFrame({col: data})
            sns.boxplot(data=box_data, y=col, ax=ax)
            ax.set_title(f'{col} Box Plot')
            
            # Add statistics
            stats_text = f'Mean: {data.mean():.2f}\n'
            stats_text += f'Median: {data.median():.2f}\n'
            stats_text += f'Std: {data.std():.2f}'
            ax.text(0.7, 0.5, stats_text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 3. Q-Q plot
            ax = axes[i, 2]
            from scipy import stats
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f'{col} Q-Q Plot')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Distribution Analysis', fontsize=14)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'distribution_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                               features: Optional[List[str]] = None,
                               method: str = 'pearson',
                               save: bool = True) -> plt.Figure:
        """
        Plot correlation heatmap
        
        Args:
            df: DataFrame
            features: Features to include (None for all numeric)
            method: Correlation method
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col != 'year']
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr(method=method)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(f'{method.title()} Correlation Matrix', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'correlation_heatmap_{method}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_time_series_decomposition(self, df: pd.DataFrame,
                                     team_name: str,
                                     metric: str = 'wins',
                                     save: bool = True) -> plt.Figure:
        """
        Plot time series decomposition for a team
        
        Args:
            df: DataFrame with team data
            team_name: Team to analyze
            metric: Metric to decompose
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        team_df = df[df['team_name'] == team_name].sort_values('year')
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # 1. Original series
        ax = axes[0]
        ax.plot(team_df['year'], team_df[metric], 'b-', linewidth=2)
        ax.set_title(f'{team_name} - {metric.title()} Time Series')
        ax.set_ylabel('Original')
        ax.grid(True, alpha=0.3)
        
        # 2. Trend (5-year moving average)
        ax = axes[1]
        if len(team_df) >= 5:
            trend = team_df[metric].rolling(window=5, center=True).mean()
            ax.plot(team_df['year'], trend, 'g-', linewidth=2)
        ax.set_ylabel('Trend')
        ax.grid(True, alpha=0.3)
        
        # 3. Detrended (if trend exists)
        ax = axes[2]
        if len(team_df) >= 5:
            detrended = team_df[metric] - trend
            ax.plot(team_df['year'], detrended, 'r-', linewidth=1)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('Detrended')
        ax.grid(True, alpha=0.3)
        
        # 4. Residuals histogram
        ax = axes[3]
        if len(team_df) >= 5:
            residuals = detrended.dropna()
            ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Residual Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Residual Distribution')
        
        plt.suptitle(f'Time Series Decomposition: {team_name} {metric.title()}', 
                    fontsize=14)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{team_name.replace(" ", "_")}_ts_decomp_{metric}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _get_era(self, year: int) -> str:
        """Get era for a given year"""
        for era_name, era_info in ERA_DEFINITIONS.items():
            if era_info['years'][0] <= year <= era_info['years'][1]:
                return era_name
        return 'unknown'
    
    def create_eda_summary(self, df: pd.DataFrame,
                         output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create comprehensive EDA summary
        
        Args:
            df: DataFrame to analyze
            output_path: Path to save summary
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'data_shape': df.shape,
            'date_range': (df['year'].min(), df['year'].max()),
            'n_teams': df['team_name'].nunique(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'summary_stats': {}
        }
        
        # Summary statistics for key metrics
        key_metrics = ['wins', 'losses', 'winning_percentage', 'run_differential',
                      'runs_scored', 'runs_allowed']
        
        for metric in key_metrics:
            if metric in df.columns:
                summary['summary_stats'][metric] = {
                    'mean': df[metric].mean(),
                    'std': df[metric].std(),
                    'min': df[metric].min(),
                    'max': df[metric].max(),
                    'q1': df[metric].quantile(0.25),
                    'median': df[metric].median(),
                    'q3': df[metric].quantile(0.75)
                }
        
        # Era breakdown
        if 'year' in df.columns:
            df['era'] = df['year'].apply(self._get_era)
            summary['era_breakdown'] = df['era'].value_counts().to_dict()
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        return summary


# Convenience functions
def plot_team_performance(df: pd.DataFrame, team_name: str,
                         save_dir: Optional[Path] = None) -> plt.Figure:
    """Quick function to plot team performance"""
    plotter = ExploratoryPlotter(save_dir=save_dir)
    return plotter.plot_team_performance_overview(df, team_name)


def plot_season_trends(df: pd.DataFrame, year: int,
                      save_dir: Optional[Path] = None) -> plt.Figure:
    """Plot trends for a specific season"""
    season_df = df[df['year'] == year]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Win distribution
    ax = axes[0, 0]
    ax.hist(season_df['wins'], bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(x=season_df['wins'].mean(), color='red', linestyle='--', 
               label=f'Mean: {season_df["wins"].mean():.1f}')
    ax.set_xlabel('Wins')
    ax.set_ylabel('Number of Teams')
    ax.set_title(f'{year} Win Distribution')
    ax.legend()
    
    # 2. Run differential vs wins
    ax = axes[0, 1]
    sns.scatterplot(data=season_df, x='run_differential', y='wins', ax=ax)
    ax.set_title(f'{year} Run Differential vs Wins')
    
    # 3. Top teams
    ax = axes[1, 0]
    top_teams = season_df.nlargest(10, 'wins')[['team_name', 'wins']]
    ax.barh(range(len(top_teams)), top_teams['wins'])
    ax.set_yticks(range(len(top_teams)))
    ax.set_yticklabels(top_teams['team_name'])
    ax.set_xlabel('Wins')
    ax.set_title(f'Top 10 Teams - {year}')
    
    # 4. League statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"Season {year} Statistics\n\n"
    stats_text += f"Average Wins: {season_df['wins'].mean():.1f}\n"
    stats_text += f"Avg Run Differential: {season_df['run_differential'].mean():.1f}\n"
    stats_text += f"Total Runs Scored: {season_df['runs_scored'].sum():,}\n"
    stats_text += f"Runs per Game: {season_df['runs_scored'].sum() / (season_df['wins'].sum() + season_df['losses'].sum()):.2f}"
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'{year} Season Overview', fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        ensure_dir(save_dir)
        plt.savefig(save_dir / f'season_trends_{year}.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_distributions(df: pd.DataFrame, features: List[str],
                             save_dir: Optional[Path] = None) -> plt.Figure:
    """Plot distributions of multiple features"""
    plotter = ExploratoryPlotter(save_dir=save_dir)
    return plotter.plot_distribution_analysis(df, features)


def plot_correlation_matrix(df: pd.DataFrame, features: Optional[List[str]] = None,
                          save_dir: Optional[Path] = None) -> plt.Figure:
    """Plot correlation matrix"""
    plotter = ExploratoryPlotter(save_dir=save_dir)
    return plotter.plot_correlation_heatmap(df, features)


def create_eda_report(df: pd.DataFrame, output_dir: Path,
                     team_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create comprehensive EDA report with plots
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory for output
        team_names: Specific teams to analyze
        
    Returns:
        Summary dictionary
    """
    ensure_dir(output_dir)
    plotter = ExploratoryPlotter(save_dir=output_dir)
    
    # Create summary
    summary = plotter.create_eda_summary(df, output_dir / 'eda_summary.json')
    
    # Create plots
    plots_created = []
    
    # 1. Feature distributions
    numeric_features = df.select_dtypes(include=[np.number]).columns[:6]
    fig = plotter.plot_distribution_analysis(df, numeric_features.tolist())
    plots_created.append('distribution_analysis')
    plt.close(fig)
    
    # 2. Correlation heatmap
    fig = plotter.plot_correlation_heatmap(df)
    plots_created.append('correlation_heatmap')
    plt.close(fig)
    
    # 3. Era comparison
    fig = plotter.plot_era_comparison(df)
    plots_created.append('era_comparison')
    plt.close(fig)
    
    # 4. Team-specific analysis
    if team_names:
        for team in team_names[:3]:  # Limit to 3 teams
            if team in df['team_name'].values:
                fig = plotter.plot_team_performance_overview(df, team)
                plots_created.append(f'{team}_overview')
                plt.close(fig)
    
    summary['plots_created'] = plots_created
    logger.info(f"EDA report created in {output_dir}")
    
    return summary