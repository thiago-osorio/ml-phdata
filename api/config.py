import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "model_features.json"
ZIPCODE_DATA_PATH = BASE_DIR / "data" / "zipcode_demographics.csv"

API_TITLE = "House Price Prediction API"
API_VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)