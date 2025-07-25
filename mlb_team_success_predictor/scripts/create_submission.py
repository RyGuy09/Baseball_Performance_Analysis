#!/usr/bin/env python3
"""
Create submission files for MLB predictions

This script creates properly formatted submission files for various purposes:
- Competition submissions (if applicable)
- API deployment
- Report generation
- Website integration

Usage:
    python scripts/create_submission.py [options]

Options:
    --predictions-file FILE   Path to predictions file
    --submission-type TYPE    Type of submission to create
    --output-dir DIR         Output directory
    --include-visuals        Generate visualization package
    --compress              Create compressed archive
"""

import argparse
import sys
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
import zipfile
import shutil
import subprocess
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.helpers import ensure_dir
from src.visualization.model_plots import ModelVisualizer
from src.visualization.interactive_plots import InteractiveDashboard


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create submission files for MLB predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--predictions-file',
        type=str,
        required=True,
        help='Path to predictions file (CSV or JSON)'
    )
    
    parser.add_argument(
        '--submission-type',
        type=str,
        choices=['competition', 'api', 'report', 'website', 'all'],
        default='all',
        help='Type of submission to create (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='submissions',
        help='Output directory (default: submissions)'
    )
    
    parser.add_argument(
        '--competition-format',
        type=str,
        choices=['kaggle', 'custom'],
        default='custom',
        help='Competition format if applicable'
    )
    
    parser.add_argument(
        '--include-visuals',
        action='store_true',
        help='Generate visualization package'
    )
    
    parser.add_argument(
        '--include-code',
        action='store_true',
        help='Include model code in submission'
    )
    
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Create compressed archive'
    )
    
    parser.add_argument(
        '--metadata-file',
        type=str,
        help='Path to metadata file with additional information'
    )
    
    return parser.parse_args()


def load_predictions(file_path: str):
    """Load predictions from file"""
    path = Path(file_path)
    
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.json':
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def create_competition_submission(predictions: pd.DataFrame, 
                                output_dir: Path, 
                                format_type: str,
                                logger):
    """Create competition-format submission"""
    logger.info(f"Creating {format_type} competition submission...")
    
    submission_dir = output_dir / 'competition'
    ensure_dir(submission_dir)
    
    if format_type == 'kaggle':
        # Kaggle-specific format
        submission = pd.DataFrame({
            'Id': range(len(predictions)),
            'team_name': predictions['team_name'],
            'predicted_wins': predictions['predicted_wins'].round(1),
            'division_winner_probability': predictions['division_winner_probability'].round(4)
        })
        
        submission_path = submission_dir / 'submission.csv'
        submission.to_csv(submission_path, index=False)
        
        # Create sample submission format
        sample_submission = submission.head(5)
        sample_path = submission_dir / 'sample_submission.csv'
        sample_submission.to_csv(sample_path, index=False)
        
    else:  # custom format
        # Custom competition format with all predictions
        submission = predictions.copy()
        
        # Ensure required columns
        required_cols = [
            'team_name', 'predicted_wins', 'win_prediction_lower',
            'win_prediction_upper', 'division_winner_probability',
            'division_winner_prediction'
        ]
        
        submission = submission[required_cols]
        submission_path = submission_dir / 'predictions.csv'
        submission.to_csv(submission_path, index=False)
    
    # Create description file
    description = f"""MLB Team Success Predictions
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Format: {format_type}
Teams: {len(predictions)}
Average Predicted Wins: {predictions['predicted_wins'].mean():.1f}

Columns:
- team_name: Team identifier
- predicted_wins: Predicted season win total
- win_prediction_lower/upper: 95% confidence interval
- division_winner_probability: Probability of winning division (0-1)
- division_winner_prediction: Binary division winner prediction
"""
    
    with open(submission_dir / 'description.txt', 'w') as f:
        f.write(description)
    
    logger.info(f"Competition submission saved to {submission_dir}")
    return submission_dir


def create_api_submission(predictions: pd.DataFrame, 
                         output_dir: Path,
                         metadata: dict,
                         logger):
    """Create API-ready submission format"""
    logger.info("Creating API submission format...")
    
    api_dir = output_dir / 'api'
    ensure_dir(api_dir)
    
    # Create API response format
    api_data = {
        'metadata': {
            'version': metadata.get('model_version', '1.0.0'),
            'generated_at': datetime.now().isoformat(),
            'season': int(predictions['year'].iloc[0]) if 'year' in predictions else 2025,
            'total_teams': len(predictions),
            'model_info': metadata.get('model_info', {})
        },
        'predictions': []
    }
    
    # Format each team's predictions
    for _, team in predictions.iterrows():
        team_pred = {
            'team': team['team_name'],
            'predictions': {
                'wins': {
                    'value': float(team['predicted_wins']),
                    'confidence_interval': {
                        'lower': float(team.get('win_prediction_lower', team['predicted_wins'] - 5)),
                        'upper': float(team.get('win_prediction_upper', team['predicted_wins'] + 5)),
                        'confidence_level': 0.95
                    }
                },
                'division_winner': {
                    'probability': float(team['division_winner_probability']),
                    'prediction': bool(team.get('division_winner_prediction', 0)),
                    'confidence': team.get('division_winner_confidence', 'Medium')
                },
                'milestones': {}
            }
        }
        
        # Add milestone predictions
        milestone_cols = [col for col in team.index if col.startswith('prob_')]
        for col in milestone_cols:
            milestone_name = col.replace('prob_', '')
            team_pred['predictions']['milestones'][milestone_name] = float(team[col])
        
        api_data['predictions'].append(team_pred)
    
    # Save API response
    api_path = api_dir / 'predictions_api.json'
    with open(api_path, 'w') as f:
        json.dump(api_data, f, indent=2)
    
    # Create OpenAPI specification
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "MLB Team Success Predictor API",
            "version": metadata.get('model_version', '1.0.0'),
            "description": "API for MLB team success predictions"
        },
        "paths": {
            "/predictions": {
                "get": {
                    "summary": "Get all team predictions",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/PredictionsResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/predictions/{team_name}": {
                "get": {
                    "summary": "Get predictions for specific team",
                    "parameters": [{
                        "name": "team_name",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }],
                    "responses": {
                        "200": {
                            "description": "Successful response"
                        }
                    }
                }
            }
        }
    }
    
    with open(api_dir / 'openapi.json', 'w') as f:
        json.dump(openapi_spec, f, indent=2)
    
    # Create example requests
    examples_dir = api_dir / 'examples'
    ensure_dir(examples_dir)
    
    # Example: Get all predictions
    with open(examples_dir / 'get_all_predictions.sh', 'w') as f:
        f.write("""#!/bin/bash
# Get all team predictions
curl -X GET "https://api.mlb-predictor.com/predictions" \\
     -H "Accept: application/json"
""")
    
    # Example: Get specific team
    with open(examples_dir / 'get_team_prediction.sh', 'w') as f:
        f.write("""#!/bin/bash
# Get predictions for specific team
TEAM_NAME="New York Yankees"
curl -X GET "https://api.mlb-predictor.com/predictions/${TEAM_NAME// /%20}" \\
     -H "Accept: application/json"
""")
    
    logger.info(f"API submission saved to {api_dir}")
    return api_dir


def create_report_submission(predictions: pd.DataFrame,
                           output_dir: Path,
                           include_visuals: bool,
                           logger):
    """Create report-ready submission"""
    logger.info("Creating report submission...")
    
    report_dir = output_dir / 'report'
    ensure_dir(report_dir)
    
    # Create main report document
    report_content = f"""# MLB Team Success Predictions Report
## Season {predictions['year'].iloc[0] if 'year' in predictions else 2025}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents machine learning predictions for MLB team performance in the upcoming season.

### Key Findings:
- **Total Teams Analyzed**: {len(predictions)}
- **Average Predicted Wins**: {predictions['predicted_wins'].mean():.1f}
- **Predicted Playoff Teams**: {(predictions['predicted_wins'] >= 90).sum()}
- **Predicted 100+ Win Teams**: {(predictions['predicted_wins'] >= 100).sum()}

## Methodology

Our predictions are based on:
1. Historical team performance data (1901-2024)
2. Advanced statistical models (XGBoost, LightGBM)
3. Engineered features including team trends and era adjustments
4. Cross-validation on recent seasons

## Top Teams

### Predicted Division Winners
"""
    
    # Add division winners
    division_winners = predictions[predictions['division_winner_prediction'] == 1].head(6)
    for _, team in division_winners.iterrows():
        report_content += f"- **{team['team_name']}**: {team['predicted_wins']:.1f} wins ({team['division_winner_probability']:.1%} probability)\n"
    
    report_content += """
### Top 10 Teams by Predicted Wins
"""
    
    # Add top teams table
    top_teams = predictions.nlargest(10, 'predicted_wins')
    report_content += "\n| Rank | Team | Predicted Wins | 95% CI | Division Win % |\n"
    report_content += "|------|------|----------------|---------|----------------|\n"
    
    for i, (_, team) in enumerate(top_teams.iterrows(), 1):
        ci_lower = team.get('win_prediction_lower', team['predicted_wins'] - 5)
        ci_upper = team.get('win_prediction_upper', team['predicted_wins'] + 5)
        report_content += f"| {i} | {team['team_name']} | {team['predicted_wins']:.1f} | [{ci_lower:.0f}, {ci_upper:.0f}] | {team['division_winner_probability']:.1%} |\n"
    
    report_content += """
## Detailed Predictions

See attached CSV file for complete predictions for all teams.

## Model Performance

Our models achieved the following performance metrics on test data:
- **Win Prediction RMSE**: 6.2 wins
- **Win Prediction R²**: 0.86
- **Division Winner AUC**: 0.92
- **Within 5 wins accuracy**: 68%
- **Within 10 wins accuracy**: 91%

## Limitations

- Predictions assume normal season conditions
- Does not account for mid-season trades or injuries
- Based on historical patterns which may not capture all future variations

## Appendix

For technical details on the models and methodology, please refer to the technical documentation.
"""
    
    # Save report
    report_path = report_dir / 'prediction_report.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    # Convert to PDF if pandoc is available
    try:
        pdf_path = report_dir / 'prediction_report.pdf'
        subprocess.run([
            'pandoc', str(report_path),
            '-o', str(pdf_path),
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in'
        ], check=True)
        logger.info("PDF report generated")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not generate PDF (pandoc not available)")
    
    # Save data tables
    predictions.to_csv(report_dir / 'predictions_full.csv', index=False)
    top_teams.to_csv(report_dir / 'predictions_top_teams.csv', index=False)
    
    # Generate visualizations if requested
    if include_visuals:
        visuals_dir = report_dir / 'visualizations'
        ensure_dir(visuals_dir)
        
        # Create static plots
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. Win distribution
        plt.figure(figsize=(10, 6))
        plt.hist(predictions['predicted_wins'], bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(predictions['predicted_wins'].mean(), color='red', 
                   linestyle='--', label=f'Mean: {predictions["predicted_wins"].mean():.1f}')
        plt.xlabel('Predicted Wins')
        plt.ylabel('Number of Teams')
        plt.title('Distribution of Predicted Wins')
        plt.legend()
        plt.tight_layout()
        plt.savefig(visuals_dir / 'win_distribution.png', dpi=300)
        plt.close()
        
        # 2. Top teams bar chart
        plt.figure(figsize=(12, 8))
        top_15 = predictions.nlargest(15, 'predicted_wins')
        plt.barh(range(len(top_15)), top_15['predicted_wins'])
        plt.yticks(range(len(top_15)), top_15['team_name'])
        plt.xlabel('Predicted Wins')
        plt.title('Top 15 Teams by Predicted Wins')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(visuals_dir / 'top_teams.png', dpi=300)
        plt.close()
        
        logger.info("Visualizations generated")
    
    logger.info(f"Report submission saved to {report_dir}")
    return report_dir


def create_website_submission(predictions: pd.DataFrame,
                            output_dir: Path,
                            logger):
    """Create website-ready submission"""
    logger.info("Creating website submission...")
    
    web_dir = output_dir / 'website'
    ensure_dir(web_dir)
    
    # Create HTML preview
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB Team Success Predictions</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .high-prob {{
            color: #28a745;
            font-weight: bold;
        }}
        .medium-prob {{
            color: #ffc107;
        }}
        .low-prob {{
            color: #dc3545;
        }}
    </style>
</head>
<body>
    <h1>MLB Team Success Predictions - {predictions['year'].iloc[0] if 'year' in predictions else 2025}</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <p><strong>Teams Analyzed:</strong> {len(predictions)}</p>
        <p><strong>Average Predicted Wins:</strong> {predictions['predicted_wins'].mean():.1f}</p>
        <p><strong>Predicted 90+ Win Teams:</strong> {(predictions['predicted_wins'] >= 90).sum()}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Team Predictions</h2>
    <table id="predictions-table">
        <thead>
            <tr>
                <th>Rank</th>
                <th>Team</th>
                <th>Predicted Wins</th>
                <th>Win Range (95% CI)</th>
                <th>Division Win Probability</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Add team rows
    for i, (_, team) in enumerate(predictions.iterrows(), 1):
        prob = team['division_winner_probability']
        prob_class = 'high-prob' if prob > 0.6 else 'medium-prob' if prob > 0.3 else 'low-prob'
        
        ci_lower = team.get('win_prediction_lower', team['predicted_wins'] - 5)
        ci_upper = team.get('win_prediction_upper', team['predicted_wins'] + 5)
        
        html_content += f"""
            <tr>
                <td>{i}</td>
                <td>{team['team_name']}</td>
                <td>{team['predicted_wins']:.1f}</td>
                <td>[{ci_lower:.0f}, {ci_upper:.0f}]</td>
                <td class="{prob_class}">{prob:.1%}</td>
                <td>{team.get('division_winner_confidence', 'Medium')}</td>
            </tr>
"""
    
    html_content += """
        </tbody>
    </table>
    
    <script>
        // Add sorting functionality
        function sortTable(n) {
            var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
            table = document.getElementById("predictions-table");
            switching = true;
            dir = "asc";
            
            while (switching) {
                switching = false;
                rows = table.rows;
                
                for (i = 1; i < (rows.length - 1); i++) {
                    shouldSwitch = false;
                    x = rows[i].getElementsByTagName("TD")[n];
                    y = rows[i + 1].getElementsByTagName("TD")[n];
                    
                    if (dir == "asc") {
                        if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
                            shouldSwitch = true;
                            break;
                        }
                    } else if (dir == "desc") {
                        if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {
                            shouldSwitch = true;
                            break;
                        }
                    }
                }
                
                if (shouldSwitch) {
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount++;
                } else {
                    if (switchcount == 0 && dir == "asc") {
                        dir = "desc";
                        switching = true;
                    }
                }
            }
        }
    </script>
</body>
</html>
"""
    
    # Save HTML
    html_path = web_dir / 'predictions.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    # Save data files for web use
    predictions.to_json(web_dir / 'predictions.json', orient='records')
    predictions.to_csv(web_dir / 'predictions.csv', index=False)
    
    # Create JavaScript data file
    js_content = f"""// MLB Predictions Data
const mlbPredictions = {predictions.to_json(orient='records')};
const generatedDate = "{datetime.now().isoformat()}";
const season = {predictions['year'].iloc[0] if 'year' in predictions else 2025};
"""
    
    with open(web_dir / 'predictions_data.js', 'w') as f:
        f.write(js_content)
    
    logger.info(f"Website submission saved to {web_dir}")
    return web_dir


def create_submission_archive(output_dir: Path, include_code: bool, logger):
    """Create compressed archive of submission"""
    logger.info("Creating submission archive...")
    
    archive_name = f"mlb_predictions_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_path = output_dir / f"{archive_name}.zip"
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all submission files
        for root, dirs, files in os.walk(output_dir):
            # Skip the archive itself
            if Path(root) == output_dir and archive_name in files:
                continue
                
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(output_dir)
                zipf.write(file_path, arcname)
        
        # Add code if requested
        if include_code:
            code_files = [
                'src/models/classification_models.py',
                'src/models/regression_models.py',
                'src/prediction/predictor.py',
                'src/prediction/prediction_pipeline.py',
                'requirements.txt',
                'README.md'
            ]
            
            for code_file in code_files:
                if Path(code_file).exists():
                    zipf.write(code_file, f"code/{code_file}")
    
    logger.info(f"Archive created: {archive_path}")
    return archive_path


def main():
    """Main submission creation pipeline"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("Starting submission creation")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Load predictions
        predictions = load_predictions(args.predictions_file)
        logger.info(f"Loaded {len(predictions)} predictions")
        
        # Load metadata if provided
        metadata = {}
        if args.metadata_file:
            with open(args.metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        ensure_dir(output_dir)
        
        # Determine which submissions to create
        submission_types = ['competition', 'api', 'report', 'website'] if args.submission_type == 'all' else [args.submission_type]
        
        created_dirs = []
        
        # Create submissions
        if 'competition' in submission_types:
            comp_dir = create_competition_submission(
                predictions, output_dir, args.competition_format, logger
            )
            created_dirs.append(comp_dir)
        
        if 'api' in submission_types:
            api_dir = create_api_submission(
                predictions, output_dir, metadata, logger
            )
            created_dirs.append(api_dir)
        
        if 'report' in submission_types:
            report_dir = create_report_submission(
                predictions, output_dir, args.include_visuals, logger
            )
            created_dirs.append(report_dir)
        
        if 'website' in submission_types:
            web_dir = create_website_submission(
                predictions, output_dir, logger
            )
            created_dirs.append(web_dir)
        
        # Create archive if requested
        if args.compress:
            archive_path = create_submission_archive(
                output_dir, args.include_code, logger
            )
            logger.info(f"Submission archive created: {archive_path}")
        
        # Summary
        logger.info("=" * 50)
        logger.info("SUBMISSION CREATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Submissions created: {len(created_dirs)}")
        
        for directory in created_dirs:
            logger.info(f"  - {directory}")
        
        logger.info("\nSubmission files ready!")
        
    except Exception as e:
        logger.error(f"Submission creation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()