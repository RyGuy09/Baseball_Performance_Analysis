#!/usr/bin/env python3
"""
Generate predictions using trained MLB models

This script loads trained models and generates predictions for specified teams/seasons.

Usage:
    python scripts/generate_predictions.py [options]

Options:
    --year YEAR              Year to predict (default: 2025)
    --teams TEAM1 TEAM2      Specific teams to predict (default: all)
    --output-dir DIR         Output directory for predictions
    --format {json,csv,all}  Output format
    --include-confidence     Include confidence intervals
    --verbose               Enable verbose output
"""

import argparse
import sys
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.prediction.prediction_pipeline import PredictionPipeline
from src.data.data_loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.utils.helpers import ensure_dir
from src.visualization.interactive_plots import InteractiveDashboard


def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate MLB predictions using trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=2025,
        help='Year to predict (default: 2025)'
    )
    
    parser.add_argument(
        '--teams',
        type=str,
        nargs='+',
        default=None,
        help='Specific teams to predict (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='predictions',
        help='Output directory for predictions (default: predictions)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'excel', 'all'],
        default='all',
        help='Output format (default: all)'
    )
    
    parser.add_argument(
        '--include-confidence',
        action='store_true',
        help='Include confidence intervals and probabilities'
    )
    
    parser.add_argument(
        '--create-dashboard',
        action='store_true',
        help='Create interactive dashboard'
    )
    
    parser.add_argument(
        '--data-source',
        type=str,
        choices=['latest', 'historical', 'custom'],
        default='latest',
        help='Data source for current season statistics'
    )
    
    parser.add_argument(
        '--custom-data',
        type=str,
        help='Path to custom data file (if data-source is custom)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def load_season_data(year: int, data_source: str, custom_path: str = None, logger = None):
    """Load data for the season to predict"""
    if logger:
        logger.info(f"Loading data for {year} season (source: {data_source})")
    
    if data_source == 'custom' and custom_path:
        # Load custom data
        df = pd.read_csv(custom_path)
        if logger:
            logger.info(f"Loaded custom data from {custom_path}")
    else:
        # Load historical data
        loader = DataLoader()
        df = loader.load_and_validate()
        
        if data_source == 'latest':
            # Use most recent season as proxy for current season
            latest_year = df['year'].max()
            df = df[df['year'] == latest_year].copy()
            if logger:
                logger.info(f"Using {latest_year} data as proxy for {year}")
        elif data_source == 'historical':
            # Use specific historical year
            if year in df['year'].values:
                df = df[df['year'] == year].copy()
            else:
                # Use most recent available
                df = df[df['year'] == df['year'].max()].copy()
    
    # Update year to prediction year
    df['year'] = year
    
    return df


def generate_team_report(team_name: str, prediction: dict, output_dir: Path):
    """Generate detailed report for a single team"""
    report = f"""# {team_name} - {prediction.get('year', 2025)} Season Prediction Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Win Projection
- **Predicted Wins**: {prediction['predicted_wins']:.1f}
- **95% Confidence Interval**: [{prediction['win_prediction_lower']:.1f}, {prediction['win_prediction_upper']:.1f}]
- **Projected Record**: {int(prediction['predicted_wins'])}-{162 - int(prediction['predicted_wins'])}

## Division Championship
- **Win Division Probability**: {prediction['division_winner_probability']:.1%}
- **Prediction**: {'Yes' if prediction['division_winner_prediction'] else 'No'}
- **Confidence Level**: {prediction['division_winner_confidence']}

## Milestone Probabilities
"""
    
    # Add milestone probabilities
    milestone_probs = {k: v for k, v in prediction.items() if k.startswith('prob_')}
    for milestone, prob in milestone_probs.items():
        milestone_name = milestone.replace('prob_', '').replace('_', ' ').title()
        report += f"- **{milestone_name}**: {prob:.1%}\n"
    
    report += f"""
## Analysis Notes
"""
    
    # Add analysis based on predictions
    if prediction['predicted_wins'] >= 95:
        report += "- Elite team projection with strong championship potential\n"
    elif prediction['predicted_wins'] >= 90:
        report += "- Solid playoff contender with division title possibilities\n"
    elif prediction['predicted_wins'] >= 85:
        report += "- Wild card contender in competitive position\n"
    elif prediction['predicted_wins'] >= 81:
        report += "- Above .500 team with outside playoff chances\n"
    else:
        report += "- Rebuilding phase with focus on development\n"
    
    if prediction['division_winner_probability'] > 0.5:
        report += "- Favored to win division\n"
    elif prediction['division_winner_probability'] > 0.25:
        report += "- Strong division contender\n"
    
    report += f"""
---
*This prediction is based on statistical models and historical patterns. Actual results may vary due to injuries, trades, and other factors not captured in the model.*
"""
    
    # Save report
    report_path = output_dir / 'team_reports' / f"{team_name.replace(' ', '_')}_report.md"
    ensure_dir(report_path.parent)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report_path


def save_predictions(predictions: pd.DataFrame, output_dir: Path, 
                    formats: list, logger = None):
    """Save predictions in specified formats"""
    ensure_dir(output_dir)
    
    saved_files = []
    
    # Save as CSV
    if 'csv' in formats or 'all' in formats:
        csv_path = output_dir / f"mlb_predictions_{predictions['year'].iloc[0]}.csv"
        predictions.to_csv(csv_path, index=False)
        saved_files.append(csv_path)
        if logger:
            logger.info(f"Saved CSV: {csv_path}")
    
    # Save as JSON
    if 'json' in formats or 'all' in formats:
        json_path = output_dir / f"mlb_predictions_{predictions['year'].iloc[0]}.json"
        predictions.to_json(json_path, orient='records', indent=2)
        saved_files.append(json_path)
        if logger:
            logger.info(f"Saved JSON: {json_path}")
    
    # Save as Excel with formatting
    if 'excel' in formats or 'all' in formats:
        excel_path = output_dir / f"mlb_predictions_{predictions['year'].iloc[0]}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main predictions
            predictions.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Summary statistics
            summary_data = {
                'Metric': [
                    'Total Teams',
                    'Average Predicted Wins',
                    'Std Dev Wins',
                    'Teams >= 90 Wins',
                    'Teams >= 100 Wins',
                    'Teams < 81 Wins'
                ],
                'Value': [
                    len(predictions),
                    predictions['predicted_wins'].mean(),
                    predictions['predicted_wins'].std(),
                    (predictions['predicted_wins'] >= 90).sum(),
                    (predictions['predicted_wins'] >= 100).sum(),
                    (predictions['predicted_wins'] < 81).sum()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Top teams
            top_teams = predictions.nlargest(10, 'predicted_wins')[
                ['team_name', 'predicted_wins', 'division_winner_probability']
            ]
            top_teams.to_excel(writer, sheet_name='Top 10 Teams', index=False)
        
        saved_files.append(excel_path)
        if logger:
            logger.info(f"Saved Excel: {excel_path}")
    
    return saved_files


def main():
    """Main prediction generation pipeline"""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    logger.info("Starting prediction generation")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Initialize prediction pipeline
        logger.info("Loading models...")
        pipeline = PredictionPipeline()
        pipeline.load_models()
        
        # Load season data
        season_data = load_season_data(
            args.year, 
            args.data_source,
            args.custom_data,
            logger
        )
        
        # Filter teams if specified
        if args.teams:
            season_data = season_data[season_data['team_name'].isin(args.teams)]
            logger.info(f"Filtered to {len(season_data)} teams: {args.teams}")
        
        logger.info(f"Generating predictions for {len(season_data)} teams...")
        
        # Generate predictions
        predictions = pipeline.predict_season(
            season_data,
            include_confidence=args.include_confidence,
            include_milestones=True
        )
        
        # Add year column
        predictions['year'] = args.year
        
        # Sort by predicted wins
        predictions = predictions.sort_values('predicted_wins', ascending=False)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        ensure_dir(output_dir)
        
        # Save predictions
        formats = [args.format] if args.format != 'all' else ['csv', 'json', 'excel']
        saved_files = save_predictions(predictions, output_dir, formats, logger)
        
        # Generate team reports for top teams
        logger.info("Generating team reports...")
        top_teams = predictions.head(10)
        
        for _, team_pred in top_teams.iterrows():
            report_path = generate_team_report(
                team_pred['team_name'],
                team_pred.to_dict(),
                output_dir
            )
            logger.debug(f"Generated report: {report_path}")
        
        # Create interactive dashboard if requested
        if args.create_dashboard:
            logger.info("Creating interactive dashboard...")
            dashboard = InteractiveDashboard(save_dir=output_dir / 'dashboards')
            
            # Create prediction dashboard
            fig = dashboard.create_prediction_results_dashboard(
                predictions[['team_name', 'predicted_wins']].rename(
                    columns={'predicted_wins': 'wins_actual'}
                ),
                predictions[['team_name', 'predicted_wins']].rename(
                    columns={'predicted_wins': 'wins_pred'}
                ),
                model_name=f'{args.year} Season Predictions'
            )
            logger.info(f"Dashboard saved to {output_dir / 'dashboards'}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Year: {args.year}")
        logger.info(f"Teams predicted: {len(predictions)}")
        logger.info(f"Average predicted wins: {predictions['predicted_wins'].mean():.1f}")
        logger.info(f"Predicted division winners: {predictions['division_winner_prediction'].sum()}")
        
        logger.info("\nTop 5 Teams:")
        for i, (_, team) in enumerate(predictions.head(5).iterrows(), 1):
            logger.info(f"{i}. {team['team_name']}: {team['predicted_wins']:.1f} wins "
                       f"({team['division_winner_probability']:.1%} division win prob)")
        
        logger.info(f"\nPredictions saved to: {output_dir}")
        logger.info("Prediction generation complete!")
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()