"""
Setup script for MLB Team Success Predictor

This script installs the MLB Team Success Predictor package and its dependencies.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()
    # Remove comments and empty lines
    required = [line for line in required if line and not line.startswith('#')]

setup(
    name="mlb-team-success-predictor",
    version="1.0.0",
    author="MLB Analytics Team",
    author_email="analytics@mlbpredictor.com",
    description="A comprehensive machine learning system for predicting MLB team success",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlb-team-success-predictor",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/mlb-team-success-predictor/issues",
        "Documentation": "https://mlb-team-success-predictor.readthedocs.io",
        "Source Code": "https://github.com/yourusername/mlb-team-success-predictor",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=required,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
        ],
        "viz": [
            "plotly>=5.18.0",
            "seaborn>=0.13.0",
        ],
        "api": [
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlb-train=scripts.train_all_models:main",
            "mlb-predict=scripts.generate_predictions:main",
            "mlb-app=app.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    zip_safe=False,
    keywords=[
        "baseball",
        "mlb",
        "machine learning",
        "prediction",
        "sports analytics",
        "data science",
    ],
)