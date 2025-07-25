"""
Constants for MLB Team Success Predictor
"""

from typing import Dict, List, Tuple

# Current season
CURRENT_SEASON = 2024

# MLB seasons info
GAMES_PER_SEASON = {
    'modern': 162,      # 1961-present (except shortened seasons)
    'classic': 154,     # 1904-1960
    'early': 140        # Pre-1904
}

# Shortened seasons
SHORTENED_SEASONS = {
    1981: 110,  # Strike
    1994: 113,  # Strike  
    1995: 144,  # Strike
    2020: 60    # COVID-19
}

# Team name mappings (handle relocations and name changes)
TEAM_MAPPINGS = {
    # Current to historical
    'Arizona Diamondbacks': ['Arizona Diamondbacks'],
    'Atlanta Braves': ['Atlanta Braves', 'Milwaukee Braves', 'Boston Braves'],
    'Baltimore Orioles': ['Baltimore Orioles', 'St. Louis Browns'],
    'Boston Red Sox': ['Boston Red Sox', 'Boston Americans'],
    'Chicago Cubs': ['Chicago Cubs', 'Chicago Orphans', 'Chicago Colts'],
    'Chicago White Sox': ['Chicago White Sox', 'Chicago White Stockings'],
    'Cincinnati Reds': ['Cincinnati Reds', 'Cincinnati Redlegs', 'Cincinnati Red Stockings'],
    'Cleveland Guardians': ['Cleveland Guardians', 'Cleveland Indians', 'Cleveland Naps'],
    'Colorado Rockies': ['Colorado Rockies'],
    'Detroit Tigers': ['Detroit Tigers'],
    'Houston Astros': ['Houston Astros', 'Houston Colt .45s'],
    'Kansas City Royals': ['Kansas City Royals'],
    'Los Angeles Angels': ['Los Angeles Angels', 'California Angels', 'Anaheim Angels'],
    'Los Angeles Dodgers': ['Los Angeles Dodgers', 'Brooklyn Dodgers', 'Brooklyn Robins'],
    'Miami Marlins': ['Miami Marlins', 'Florida Marlins'],
    'Milwaukee Brewers': ['Milwaukee Brewers', 'Seattle Pilots'],
    'Minnesota Twins': ['Minnesota Twins', 'Washington Senators'],
    'New York Mets': ['New York Mets'],
    'New York Yankees': ['New York Yankees', 'New York Highlanders'],
    'Oakland Athletics': ['Oakland Athletics', 'Kansas City Athletics', 'Philadelphia Athletics'],
    'Philadelphia Phillies': ['Philadelphia Phillies'],
    'Pittsburgh Pirates': ['Pittsburgh Pirates'],
    'San Diego Padres': ['San Diego Padres'],
    'San Francisco Giants': ['San Francisco Giants', 'New York Giants'],
    'Seattle Mariners': ['Seattle Mariners'],
    'St. Louis Cardinals': ['St. Louis Cardinals', 'St. Louis Browns', 'St. Louis Perfectos'],
    'Tampa Bay Rays': ['Tampa Bay Rays', 'Tampa Bay Devil Rays'],
    'Texas Rangers': ['Texas Rangers', 'Washington Senators'],
    'Toronto Blue Jays': ['Toronto Blue Jays'],
    'Washington Nationals': ['Washington Nationals', 'Montreal Expos']
}

# Reverse mapping
HISTORICAL_TO_CURRENT = {}
for current, historical_list in TEAM_MAPPINGS.items():
    for historical in historical_list:
        HISTORICAL_TO_CURRENT[historical] = current

# Era definitions
ERA_DEFINITIONS = {
    'dead_ball': {
        'years': (1901, 1919),
        'description': 'Dead Ball Era - Low scoring, emphasis on small ball',
        'avg_runs_per_game': 3.9
    },
    'live_ball': {
        'years': (1920, 1941),
        'description': 'Live Ball Era - Introduction of livelier ball, rise of home runs',
        'avg_runs_per_game': 5.0
    },
    'integration': {
        'years': (1942, 1962),
        'description': 'Integration Era - Breaking of color barrier, expansion',
        'avg_runs_per_game': 4.5
    },
    'expansion': {
        'years': (1963, 1976),
        'description': 'Expansion Era - League expansion, pitcher dominance',
        'avg_runs_per_game': 4.0
    },
    'free_agency': {
        'years': (1977, 1993),
        'description': 'Free Agency Era - Player movement, competitive balance',
        'avg_runs_per_game': 4.3
    },
    'steroid': {
        'years': (1994, 2005),
        'description': 'Steroid Era - High offense, power hitting',
        'avg_runs_per_game': 4.8
    },
    'modern': {
        'years': (2006, CURRENT_SEASON),
        'description': 'Modern Era - Analytics, shifts, three true outcomes',
        'avg_runs_per_game': 4.4
    }
}

# Milestone thresholds
MILESTONE_THRESHOLDS = {
    'wins': {
        'terrible': 65,
        'poor': 75,
        'average': 81,
        'good': 90,
        'excellent': 95,
        'elite': 100,
        'historic': 105
    },
    'runs_scored': {
        'low': 650,
        'below_average': 700,
        'average': 750,
        'above_average': 800,
        'high': 850,
        'elite': 900
    },
    'runs_allowed': {
        'elite': 600,
        'excellent': 650,
        'good': 700,
        'average': 750,
        'below_average': 800,
        'poor': 850
    },
    'run_differential': {
        'terrible': -150,
        'poor': -75,
        'below_average': -25,
        'average': 0,
        'above_average': 50,
        'good': 100,
        'excellent': 150,
        'elite': 200
    }
}

# Division and league structure
DIVISIONS = {
    'AL East': ['Baltimore Orioles', 'Boston Red Sox', 'New York Yankees', 
                'Tampa Bay Rays', 'Toronto Blue Jays'],
    'AL Central': ['Chicago White Sox', 'Cleveland Guardians', 'Detroit Tigers',
                   'Kansas City Royals', 'Minnesota Twins'],
    'AL West': ['Houston Astros', 'Los Angeles Angels', 'Oakland Athletics',
                'Seattle Mariners', 'Texas Rangers'],
    'NL East': ['Atlanta Braves', 'Miami Marlins', 'New York Mets',
                'Philadelphia Phillies', 'Washington Nationals'],
    'NL Central': ['Chicago Cubs', 'Cincinnati Reds', 'Milwaukee Brewers',
                   'Pittsburgh Pirates', 'St. Louis Cardinals'],
    'NL West': ['Arizona Diamondbacks', 'Colorado Rockies', 'Los Angeles Dodgers',
                'San Diego Padres', 'San Francisco Giants']
}

# League mapping
LEAGUES = {
    'American League': ['AL East', 'AL Central', 'AL West'],
    'National League': ['NL East', 'NL Central', 'NL West']
}

# Playoff structure by era
PLAYOFF_STRUCTURE = {
    'pre_division': {
        'years': (1901, 1968),
        'teams': 2,
        'description': 'Only World Series (best AL vs best NL)'
    },
    'division_era': {
        'years': (1969, 1993),
        'teams': 4,
        'description': 'Division winners only'
    },
    'wild_card_era': {
        'years': (1994, 2011),
        'teams': 8,
        'description': 'Division winners + 1 wild card per league'
    },
    'expanded_wild_card': {
        'years': (2012, 2021),
        'teams': 10,
        'description': 'Division winners + 2 wild cards per league'
    },
    'expanded_playoffs': {
        'years': (2022, CURRENT_SEASON),
        'teams': 12,
        'description': 'Division winners + 3 wild cards per league'
    }
}

# Statistical categories
BATTING_STATS = [
    'AVG', 'OBP', 'SLG', 'OPS', 'HR', 'RBI', 'R', 'H', '2B', '3B', 'BB', 'SO'
]

PITCHING_STATS = [
    'ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'W', 'L', 'SV', 'IP', 'SO', 'BB'
]

ADVANCED_STATS = [
    'wRC+', 'WAR', 'FIP', 'xFIP', 'BABIP', 'wOBA', 'ISO', 'K%', 'BB%'
]

# Model type mappings
MODEL_TYPE_MAPPING = {
    'classification': {
        'division_winner': 'DivisionWinnerClassifier',
        'playoff': 'PlayoffClassifier',
        'milestone': 'MilestonePredictor'
    },
    'regression': {
        'wins': 'WinsRegressor',
        'runs_scored': 'RunProductionRegressor',
        'runs_allowed': 'RunProductionRegressor'
    },
    'time_series': {
        'forecast': 'TeamPerformanceForecaster',
        'decompose': 'SeasonalDecomposer'
    }
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    'min_games': 60,  # Minimum games for valid season
    'max_wins': 120,  # Maximum realistic wins (for data validation)
    'min_year': 1901,  # Start of modern era
    'max_year': CURRENT_SEASON + 1  # Allow for next season predictions
}

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    'classification': {
        'excellent': {'roc_auc': 0.90, 'accuracy': 0.85},
        'good': {'roc_auc': 0.80, 'accuracy': 0.75},
        'acceptable': {'roc_auc': 0.70, 'accuracy': 0.65}
    },
    'regression': {
        'excellent': {'rmse': 5.0, 'r2': 0.80},
        'good': {'rmse': 8.0, 'r2': 0.65},
        'acceptable': {'rmse': 12.0, 'r2': 0.50}
    }
}


# Utility functions
def get_era(year: int) -> str:
    """Get era name for a given year"""
    for era_name, era_info in ERA_DEFINITIONS.items():
        start_year, end_year = era_info['years']
        if start_year <= year <= end_year:
            return era_name
    return 'unknown'

def get_current_team_name(historical_name: str) -> str:
    """Get current team name from historical name"""
    return HISTORICAL_TO_CURRENT.get(historical_name, historical_name)

def get_division(team_name: str) -> str:
    """Get division for a team"""
    current_name = get_current_team_name(team_name)
    for division, teams in DIVISIONS.items():
        if current_name in teams:
            return division
    return 'unknown'

def get_league(team_name: str) -> str:
    """Get league for a team"""
    division = get_division(team_name)
    for league, divisions in LEAGUES.items():
        if division in divisions:
            return league
    return 'unknown'

def is_shortened_season(year: int) -> bool:
    """Check if a year had a shortened season"""
    return year in SHORTENED_SEASONS

def get_expected_games(year: int) -> int:
    """Get expected number of games for a season"""
    if year in SHORTENED_SEASONS:
        return SHORTENED_SEASONS[year]
    elif year >= 1961:
        return GAMES_PER_SEASON['modern']
    elif year >= 1904:
        return GAMES_PER_SEASON['classic']
    else:
        return GAMES_PER_SEASON['early']


if __name__ == "__main__":
    # Test utilities
    print("ERA for 2020:", get_era(2020))
    print("Current name for 'Brooklyn Dodgers':", get_current_team_name('Brooklyn Dodgers'))
    print("Division for 'New York Yankees':", get_division('New York Yankees'))
    print("Is 2020 shortened season?", is_shortened_season(2020))
    print("Expected games in 1995:", get_expected_games(1995))