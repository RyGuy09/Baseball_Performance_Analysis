"""
Visualization module for MLB Team Success Predictor

This module provides:
- Exploratory data analysis plots
- Model performance visualizations
- Interactive dashboards
- Report generation
"""

from .exploratory_plots import (
    ExploratoryPlotter,
    plot_team_performance,
    plot_season_trends,
    plot_feature_distributions,
    plot_correlation_matrix,
    create_eda_report
)

from .model_plots import (
    ModelVisualizer,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_prediction_results,
    plot_residuals,
    create_model_report
)

from .interactive_plots import (
    InteractiveDashboard,
    create_team_dashboard,
    create_season_dashboard,
    create_prediction_dashboard,
    create_comparison_dashboard
)

__all__ = [
    # Exploratory plots
    'ExploratoryPlotter',
    'plot_team_performance',
    'plot_season_trends',
    'plot_feature_distributions',
    'plot_correlation_matrix',
    'create_eda_report',
    
    # Model plots
    'ModelVisualizer',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_feature_importance',
    'plot_prediction_results',
    'plot_residuals',
    'create_model_report',
    
    # Interactive plots
    'InteractiveDashboard',
    'create_team_dashboard',
    'create_season_dashboard',
    'create_prediction_dashboard',
    'create_comparison_dashboard'
]

# Visualization settings
VIZ_CONFIG = {
    'figure_size': (10, 6),
    'dpi': 100,
    'style': 'seaborn',
    'color_palette': 'husl',
    'save_format': 'png'
}