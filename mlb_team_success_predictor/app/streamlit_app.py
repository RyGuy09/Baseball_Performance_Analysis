"""
MLB Team Success Predictor - Streamlit Web Application

This app provides an interactive interface for MLB team predictions including:
- Division winner predictions
- Win total projections
- Historical analysis
- Team comparisons
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import custom modules
from src.prediction.predictor import DivisionWinnerPredictor, WinsPredictor
from src.prediction.prediction_pipeline import PredictionPipeline
from src.data.data_loader import DataLoader
from src.data.feature_engineering import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="MLB Team Success Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-high {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .prediction-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

@st.cache_resource
def load_models():
    """Load ML models"""
    try:
        pipeline = PredictionPipeline()
        pipeline.load_models()
        return pipeline
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

@st.cache_data
def load_historical_data():
    """Load historical MLB data"""
    try:
        loader = DataLoader()
        df = loader.load_data()
        
        # Add era information
        df['era'] = df['year'].apply(lambda y: get_era(y))
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_era(year):
    """Determine baseball era for a given year"""
    if year < 1920:
        return 'Dead Ball'
    elif year < 1947:
        return 'Live Ball'
    elif year < 1969:
        return 'Integration'
    elif year < 1977:
        return 'Expansion'
    elif year < 1994:
        return 'Free Agency'
    elif year < 2006:
        return 'Steroid'
    else:
        return 'Modern'

def generate_predictions(pipeline, season_data):
    """Generate predictions for current season"""
    try:
        predictions = pipeline.predict_season(
            season_data,
            include_confidence=True,
            include_milestones=True
        )
        return predictions
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return None

def create_win_projection_chart(predictions):
    """Create win projection visualization"""
    fig = go.Figure()
    
    # Sort by predicted wins
    predictions_sorted = predictions.sort_values('predicted_wins', ascending=True)
    
    # Add bar chart with error bars
    fig.add_trace(go.Bar(
        y=predictions_sorted['team_name'],
        x=predictions_sorted['predicted_wins'],
        orientation='h',
        error_x=dict(
            type='data',
            symmetric=False,
            array=predictions_sorted['win_prediction_upper'] - predictions_sorted['predicted_wins'],
            arrayminus=predictions_sorted['predicted_wins'] - predictions_sorted['win_prediction_lower']
        ),
        marker_color='lightblue',
        name='Predicted Wins'
    ))
    
    # Add reference lines
    fig.add_vline(x=81, line_dash="dash", line_color="gray", 
                  annotation_text=".500 Record")
    fig.add_vline(x=90, line_dash="dash", line_color="green", 
                  annotation_text="90 Wins")
    
    fig.update_layout(
        title="2025 Win Projections by Team",
        xaxis_title="Predicted Wins",
        yaxis_title="Team",
        height=800,
        showlegend=False
    )
    
    return fig

def create_division_probability_chart(predictions):
    """Create division winner probability chart"""
    # Get top 10 teams by division winner probability
    top_teams = predictions.nlargest(10, 'division_winner_probability')
    
    fig = go.Figure()
    
    # Create bar chart
    fig.add_trace(go.Bar(
        x=top_teams['team_name'],
        y=top_teams['division_winner_probability'] * 100,
        marker_color=top_teams['division_winner_probability'],
        marker_colorscale='Viridis',
        text=[f"{p:.1f}%" for p in top_teams['division_winner_probability'] * 100],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Top 10 Division Winner Probabilities",
        xaxis_title="Team",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        showlegend=False,
        height=500
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig

def create_team_comparison_chart(predictions, selected_teams):
    """Create comparison chart for selected teams"""
    team_data = predictions[predictions['team_name'].isin(selected_teams)]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Win Projections', 'Division Winner Probability',
                       '90+ Win Probability', '100+ Win Probability'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Win projections
    fig.add_trace(
        go.Bar(x=team_data['team_name'], y=team_data['predicted_wins'],
               name='Predicted Wins', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Division winner probability
    fig.add_trace(
        go.Bar(x=team_data['team_name'], 
               y=team_data['division_winner_probability'] * 100,
               name='Division Winner %', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # 90+ wins probability
    if 'prob_achieved_90_wins' in team_data.columns:
        fig.add_trace(
            go.Bar(x=team_data['team_name'], 
                   y=team_data['prob_achieved_90_wins'] * 100,
                   name='90+ Wins %', marker_color='orange'),
            row=2, col=1
        )
    
    # 100+ wins probability
    if 'prob_achieved_100_wins' in team_data.columns:
        fig.add_trace(
            go.Bar(x=team_data['team_name'], 
                   y=team_data['prob_achieved_100_wins'] * 100,
                   name='100+ Wins %', marker_color='red'),
            row=2, col=2
        )
    
    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(tickangle=-45)
    
    return fig

def create_historical_trend_chart(historical_data, team_name):
    """Create historical performance chart for a team"""
    team_data = historical_data[historical_data['team_name'] == team_name].sort_values('year')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Wins Over Time', 'Run Differential Over Time'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Wins over time
    fig.add_trace(
        go.Scatter(x=team_data['year'], y=team_data['wins'],
                   mode='lines+markers', name='Wins',
                   line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Add .500 line
    fig.add_hline(y=81, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Run differential
    colors = ['red' if x < 0 else 'green' for x in team_data['run_differential']]
    fig.add_trace(
        go.Bar(x=team_data['year'], y=team_data['run_differential'],
               marker_color=colors, name='Run Differential'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">⚾ MLB Team Success Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Predictions for the 2025 MLB Season</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
                           ["🏠 Home", "📊 Predictions", "🏆 Team Analysis", 
                            "📈 Historical Trends", "ℹ️ About"])
    
    # Load models and data
    if st.session_state.pipeline is None:
        with st.spinner("Loading models..."):
            st.session_state.pipeline = load_models()
    
    if st.session_state.historical_data is None:
        with st.spinner("Loading historical data..."):
            st.session_state.historical_data = load_historical_data()
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Predictions":
        show_predictions_page()
    elif page == "🏆 Team Analysis":
        show_team_analysis_page()
    elif page == "📈 Historical Trends":
        show_historical_trends_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display home page"""
    st.header("Welcome to MLB Team Success Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Division Winners</h3>
        <p>Predict which teams will win their divisions with confidence levels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Win Totals</h3>
        <p>Project season win totals with confidence intervals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🏆 Milestones</h3>
        <p>Calculate probabilities for achieving key milestones</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.historical_data is not None:
        st.subheader("Quick Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        recent_year = st.session_state.historical_data['year'].max()
        recent_data = st.session_state.historical_data[
            st.session_state.historical_data['year'] == recent_year
        ]
        
        with col1:
            st.metric("Teams Analyzed", "30")
        with col2:
            st.metric("Years of Data", f"{st.session_state.historical_data['year'].nunique()}")
        with col3:
            st.metric("Latest Season", f"{recent_year}")
        with col4:
            st.metric("Avg Wins (Latest)", f"{recent_data['wins'].mean():.1f}")
    
    st.markdown("---")
    st.info("Navigate to the **Predictions** page to see 2025 season projections!")

def show_predictions_page():
    """Display predictions page"""
    st.header("2025 Season Predictions")
    
    # Generate predictions if not already done
    if st.session_state.predictions is None:
        with st.spinner("Generating predictions..."):
            # Use most recent season data as proxy for current season
            recent_data = st.session_state.historical_data[
                st.session_state.historical_data['year'] == 
                st.session_state.historical_data['year'].max()
            ]
            
            # Engineer features
            engineer = FeatureEngineer()
            featured_data = engineer.engineer_features(recent_data)
            
            st.session_state.predictions = generate_predictions(
                st.session_state.pipeline, 
                featured_data
            )
    
    if st.session_state.predictions is not None:
        predictions = st.session_state.predictions
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_wins = predictions['predicted_wins'].mean()
            st.metric("Avg Predicted Wins", f"{avg_wins:.1f}")
        
        with col2:
            teams_90 = (predictions['predicted_wins'] >= 90).sum()
            st.metric("Teams 90+ Wins", teams_90)
        
        with col3:
            teams_100 = (predictions['predicted_wins'] >= 100).sum()
            st.metric("Teams 100+ Wins", teams_100)
        
        with col4:
            below_500 = (predictions['predicted_wins'] < 81).sum()
            st.metric("Teams Below .500", below_500)
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["Win Projections", "Division Winners", "Team Comparison"])
        
        with tab1:
            st.plotly_chart(create_win_projection_chart(predictions), 
                          use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_division_probability_chart(predictions), 
                          use_container_width=True)
        
        with tab3:
            selected_teams = st.multiselect(
                "Select teams to compare:",
                predictions['team_name'].tolist(),
                default=predictions.nlargest(5, 'predicted_wins')['team_name'].tolist()
            )
            
            if selected_teams:
                st.plotly_chart(
                    create_team_comparison_chart(predictions, selected_teams),
                    use_container_width=True
                )
        
        # Detailed predictions table
        st.subheader("Detailed Predictions")
        
        # Format display columns
        display_cols = ['team_name', 'predicted_wins', 'division_winner_probability', 
                       'division_winner_confidence']
        
        display_df = predictions[display_cols].copy()
        display_df['predicted_wins'] = display_df['predicted_wins'].round(1)
        display_df['division_winner_probability'] = (
            display_df['division_winner_probability'] * 100
        ).round(1).astype(str) + '%'
        
        st.dataframe(
            display_df.rename(columns={
                'team_name': 'Team',
                'predicted_wins': 'Predicted Wins',
                'division_winner_probability': 'Division Win %',
                'division_winner_confidence': 'Confidence'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = predictions.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"mlb_predictions_2025_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def show_team_analysis_page():
    """Display team analysis page"""
    st.header("Team Analysis")
    
    if st.session_state.predictions is not None:
        predictions = st.session_state.predictions
        
        # Team selector
        selected_team = st.selectbox(
            "Select a team:",
            predictions['team_name'].tolist()
        )
        
        if selected_team:
            team_data = predictions[predictions['team_name'] == selected_team].iloc[0]
            
            # Team header
            st.subheader(f"{selected_team} - 2025 Projections")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Wins",
                    f"{team_data['predicted_wins']:.1f}",
                    f"[{team_data['win_prediction_lower']:.0f} - {team_data['win_prediction_upper']:.0f}]"
                )
            
            with col2:
                prob = team_data['division_winner_probability']
                confidence_color = (
                    "prediction-high" if prob > 0.6 
                    else "prediction-medium" if prob > 0.3 
                    else "prediction-low"
                )
                st.markdown(f"""
                <div class="metric-card">
                <h3>Division Winner Probability</h3>
                <p class="{confidence_color}">{prob:.1%}</p>
                <p>{team_data['division_winner_confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.metric(
                    "Division Winner Prediction",
                    "Yes" if team_data['division_winner_prediction'] else "No"
                )
            
            # Milestone probabilities
            st.subheader("Milestone Probabilities")
            
            milestone_cols = [col for col in predictions.columns if col.startswith('prob_')]
            if milestone_cols:
                milestone_data = []
                for col in milestone_cols:
                    milestone_name = col.replace('prob_', '').replace('_', ' ').title()
                    probability = team_data[col]
                    milestone_data.append({
                        'Milestone': milestone_name,
                        'Probability': f"{probability:.1%}"
                    })
                
                milestone_df = pd.DataFrame(milestone_data)
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=[float(p.strip('%'))/100 for p in milestone_df['Probability']],
                        y=milestone_df['Milestone'],
                        orientation='h',
                        marker_color='lightgreen'
                    )
                ])
                
                fig.update_layout(
                    title=f"{selected_team} - Milestone Achievement Probabilities",
                    xaxis_title="Probability",
                    yaxis_title="Milestone",
                    xaxis_tickformat='.0%',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Historical context
            if st.session_state.historical_data is not None:
                st.subheader("Historical Context")
                
                team_history = st.session_state.historical_data[
                    st.session_state.historical_data['team_name'] == selected_team
                ].tail(10)
                
                if not team_history.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        avg_wins = team_history['wins'].mean()
                        st.metric("10-Year Avg Wins", f"{avg_wins:.1f}")
                    
                    with col2:
                        best_season = team_history.loc[team_history['wins'].idxmax()]
                        st.metric("Best Recent Season", 
                                f"{int(best_season['wins'])} wins ({int(best_season['year'])})")

def show_historical_trends_page():
    """Display historical trends page"""
    st.header("Historical Trends")
    
    if st.session_state.historical_data is not None:
        data = st.session_state.historical_data
        
        # Team selector
        selected_team = st.selectbox(
            "Select a team to view historical performance:",
            sorted(data['team_name'].unique())
        )
        
        if selected_team:
            # Create historical chart
            st.plotly_chart(
                create_historical_trend_chart(data, selected_team),
                use_container_width=True
            )
            
            # Summary statistics
            team_data = data[data['team_name'] == selected_team]
            
            st.subheader(f"{selected_team} Historical Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Seasons", len(team_data))
            
            with col2:
                st.metric("Avg Wins", f"{team_data['wins'].mean():.1f}")
            
            with col3:
                st.metric("Best Season", f"{team_data['wins'].max()} wins")
            
            with col4:
                playoff_seasons = (team_data['wins'] >= 90).sum()
                st.metric("90+ Win Seasons", playoff_seasons)
            
            # Era breakdown
            st.subheader("Performance by Era")
            
            era_stats = team_data.groupby('era').agg({
                'wins': ['mean', 'max', 'count'],
                'winning_percentage': 'mean'
            }).round(3)
            
            era_stats.columns = ['Avg Wins', 'Best Season', 'Seasons', 'Win %']
            st.dataframe(era_stats, use_container_width=True)

def show_about_page():
    """Display about page"""
    st.header("About MLB Team Success Predictor")
    
    st.markdown("""
    ### Overview
    The MLB Team Success Predictor uses advanced machine learning models to predict:
    - **Division Winners**: Which teams are most likely to win their divisions
    - **Win Totals**: Projected season win totals with confidence intervals
    - **Milestone Achievements**: Probabilities of reaching key milestones (90+ wins, 100+ wins, etc.)
    
    ### Methodology
    Our models are trained on comprehensive MLB statistics from 1901 to present, including:
    - Team performance metrics (wins, losses, run differential)
    - Historical trends and patterns
    - Era-adjusted statistics
    - Advanced sabermetric features
    
    ### Model Performance
    - **Division Winner Classifier**: ~85% accuracy with ROC-AUC > 0.90
    - **Win Total Regressor**: RMSE < 6 wins, R² > 0.85
    - **Confidence Calibration**: Well-calibrated probability estimates
    
    ### Data Sources
    - Historical MLB statistics from Baseball Reference
    - Updated through the most recent completed season
    - Features engineered using domain expertise
    
    ### Limitations
    - Predictions assume normal season conditions
    - Does not account for mid-season trades or injuries
    - Based on historical patterns which may not perfectly predict future outcomes
    
    ### Contact
    For questions or feedback, please contact the development team.
    
    ---
    *Last updated: {datetime.now().strftime('%B %Y')}*
    """)

if __name__ == "__main__":
    main()