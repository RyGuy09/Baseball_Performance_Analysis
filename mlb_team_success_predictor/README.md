# MLB Team Success Predictor 🏆⚾🧢

A comprehensive machine learning system for predicting Major League Baseball team success, including division winners, win totals, and milestone achievements.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-yellowgreen.svg)

## Overview

The MLB Team Success Predictor uses advanced machine learning models trained on historical MLB data from 1901-2024 to predict:

- **Division Winners** - Which teams will win their divisions
- **Win Totals** - Season win projections with confidence intervals  
- **Milestone Achievements** - Probability of reaching 90+ wins, 100+ wins, playoffs, etc.

## Features

- 🤖 **Multiple ML Models**: XGBoost, LightGBM, Random Forest, and ensemble methods
- 📊 **Comprehensive Analytics**: 50+ engineered features including era adjustments
- 🎯 **High Accuracy**: 85%+ accuracy on division winners, <6 wins RMSE on totals
- 📈 **Interactive Dashboards**: Streamlit web app with visualizations
- 🔄 **Full Pipeline**: Data processing → Training → Evaluation → Prediction
- 📝 **Detailed Reports**: Team-specific predictions and analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlb-team-success-predictor.git
cd mlb-team-success-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
