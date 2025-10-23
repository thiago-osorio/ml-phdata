import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "model_features.json"
ZIPCODE_DATA_PATH = BASE_DIR / "data" / "zipcode_demographics.csv"
SALES_DATA_PATH = BASE_DIR / "data" / "kc_house_data.csv"
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
TARGET = "price"

API_TITLE = "House Price Prediction API"
API_VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)