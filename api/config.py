"""
Configuration settings for the House Price Prediction API.

This module contains all configuration constants, file paths, and settings
used throughout the application. It centralizes configuration management
and makes it easy to modify paths and parameters without changing code
in multiple locations.

Constants defined here include:
- File paths for models, data, and features
- Column selections for data processing
- API metadata (title, version)
- Logging configuration
"""

import logging
from pathlib import Path

# Base directory - parent of the api directory
BASE_DIR = Path(__file__).parent.parent

# Model and feature file paths
MODEL_PATH = BASE_DIR / "model" / "model.pkl"  # Main trained model file
MODEL_BACKUP_PATH = BASE_DIR / "model" / "model_backup.pkl"  # Backup model for rollback
FEATURES_PATH = BASE_DIR / "model" / "model_features.json"  # Feature names used by model

# Data file paths
ZIPCODE_DATA_PATH = BASE_DIR / "data" / "zipcode_demographics.csv"  # Demographic data by ZIP code
SALES_DATA_PATH = BASE_DIR / "data" / "kc_house_data.csv"  # House sales data

# Data processing configuration
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]  # Columns to extract from sales data
TARGET = "price"  # Target variable for machine learning

# API metadata
API_TITLE = "House Price Prediction API"
API_VERSION = "1.0.0"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)