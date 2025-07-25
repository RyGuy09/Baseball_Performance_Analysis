"""
Test suite for MLB Team Success Predictor

This package contains unit tests and integration tests for all components
of the MLB prediction system.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Test constants
TEST_RANDOM_SEED = 42
TEST_SAMPLE_SIZE = 100

__all__ = [
    'TEST_DATA_DIR',
    'TEST_OUTPUT_DIR',
    'TEST_RANDOM_SEED',
    'TEST_SAMPLE_SIZE'
]