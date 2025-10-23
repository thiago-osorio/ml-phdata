"""
Machine learning services for house price prediction.

This module contains the core business logic for training and using machine learning models
to predict house prices. It includes services for model training, prediction inference,
model management (backup/restore/retrain), and data processing.

The module provides two main service classes:
- TrainService: Handles model training and evaluation
- PredictionService: Handles prediction inference and model lifecycle management
"""

import logging
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import json
import time
import shutil
from pathlib import Path
from .config import MODEL_PATH, MODEL_BACKUP_PATH, FEATURES_PATH, ZIPCODE_DATA_PATH, SALES_DATA_PATH, SALES_COLUMN_SELECTION, TARGET

logger = logging.getLogger(__name__)


class TrainService:
    """
    Service for training machine learning models for house price prediction.

    This service handles data loading, preprocessing, model training, and model
    persistence. It combines house sales data with demographic data based on
    ZIP codes to create a comprehensive dataset for training.

    Attributes:
        data: Dictionary containing train/test splits of features and targets
    """
    def __init__(self):
        """
        Initialize the TrainService with loaded and preprocessed data.

        Loads house sales data, merges it with demographic data, and creates
        train/test splits for model training and evaluation.
        """
        logger.info("Initializing TrainService")
        self.data = self._load_data()

        logger.info("TrainService initialized successfully")

    def _load_data(self) -> Dict[str, Any]:
        """
        Load and preprocess data for model training.

        Loads house sales data and demographic data, merges them by ZIP code,
        separates features from target variable, and creates train/test splits.

        Returns:
            Dict containing train/test splits:
                - x_train: Training features
                - y_train: Training targets (prices)
                - x_test: Test features
                - y_test: Test targets (prices)
        """
        data = pd.read_csv(SALES_DATA_PATH, usecols=SALES_COLUMN_SELECTION, dtype={'zipcode': str})
        demographics = pd.read_csv(ZIPCODE_DATA_PATH,dtype={'zipcode': str})

        merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
        y = merged_data.pop('price')
        x = merged_data
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
        
        return {
            "x_train": x_train, 
            "y_train": y_train, 
            "x_test": x_test, 
            "y_test": y_test
            }

    def _train_model(self):
        """
        Train and evaluate a machine learning model for house price prediction.

        Creates a scikit-learn pipeline with RobustScaler for feature scaling
        and RandomForestRegressor for prediction. Trains the model on training
        data, evaluates it on test data, calculates MAPE (Mean Absolute Percentage Error),
        and saves the trained model to disk.

        The trained model is saved as a pickle file to MODEL_PATH.
        """
        logger.info("Starting model training")
        
        pipe = make_pipeline(RobustScaler(), RandomForestRegressor(random_state=123))
        pipe.fit(self.data["x_train"], self.data["y_train"])
        pred = pipe.predict(self.data["x_test"])
        mape = mean_absolute_percentage_error(y_true=self.data["y_test"], y_pred=pred)
        logger.info(f"MAPE {round(mape, 4)}")
        pickle.dump(pipe, open(MODEL_PATH, 'wb'))
        
        logger.info("Model trained successfully")

class PredictionService:
    """
    Service for house price predictions and model lifecycle management.

    This service handles loading trained models, making predictions on new data,
    and managing the model lifecycle including retraining, backup, and rollback
    operations. It automatically merges user-provided house features with
    demographic data based on ZIP codes.

    Attributes:
        model: Loaded scikit-learn model pipeline
        features_data: List of feature names expected by the model
        zipcode_df: DataFrame containing demographic data indexed by ZIP code
    """

    def __init__(self, model=None, features_data=None, zipcode_df=None):
        """
        Initialize the PredictionService.

        Args:
            model: Pre-loaded model (optional, will load from disk if not provided)
            features_data: Pre-loaded feature list (optional, will load from disk if not provided)
            zipcode_df: Pre-loaded demographic data (optional, will load from disk if not provided)
        """
        logger.info("Initializing PredictionService")

        if model is not None and features_data is not None and zipcode_df is not None:
            self.model = model
            self.features_data = features_data if "zipcode" in features_data else features_data + ["zipcode"]
            self.zipcode_df = zipcode_df
        else:
            self.model = self._load_model()
            self.features_data = self._load_features()
            self.zipcode_df = self._load_zipcode_data()

        logger.info("PredictionService initialized successfully")

    def reload(self):
        """
        Reload the model, features, and zipcode data from disk.

        This method refreshes all components of the prediction service by
        reloading them from their respective files. Useful after model
        retraining or configuration changes.

        Returns:
            Dict with status and message indicating success or failure
        """
        try:
            logger.info("Reloading PredictionService components")
            self.model = self._load_model()
            self.features_data = self._load_features()
            self.zipcode_df = self._load_zipcode_data()
            logger.info("PredictionService reloaded successfully")
            return {"status": "success", "message": "Model reloaded successfully"}
        except Exception as e:
            logger.error(f"Error reloading PredictionService: {e}")
            return {"status": "error", "message": str(e)}

    def _backup_model(self):
        """Create a backup of the current model."""
        try:
            if Path(MODEL_PATH).exists():
                logger.info("Creating model backup")
                shutil.copy2(MODEL_PATH, MODEL_BACKUP_PATH)
                logger.info("Model backup created successfully")
                return True
            else:
                logger.warning("No model file found to backup")
                return False
        except Exception as e:
            logger.error(f"Error creating model backup: {e}")
            return False

    def retrain(self):
        """
        Retrain the model using fresh data with automatic backup.

        Creates a backup of the current model before training a new one.
        Uses TrainService to train a new model with the latest data,
        then reloads the prediction service with the new model.

        Returns:
            Dict with status and message indicating success or failure,
            including information about backup creation status
        """
        try:
            logger.info("Starting model retraining")

            # Create backup before retraining
            backup_success = self._backup_model()
            if not backup_success:
                logger.warning("Model backup failed, continuing with retrain anyway")

            # Retrain the model
            train_service = TrainService()
            train_service._train_model()

            # Reload after training
            self.reload()
            logger.info("Model retrained and reloaded successfully")

            backup_msg = " (backup created)" if backup_success else " (backup failed - old model may not be recoverable)"
            return {"status": "success", "message": f"Model retrained successfully{backup_msg}"}

        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return {"status": "error", "message": str(e)}

    def rollback(self):
        """
        Rollback to the previous model version from backup.

        Restores the model from the backup file created during the last
        retraining operation. This allows quick recovery if a new model
        performs poorly or has issues.

        Returns:
            Dict with status and message indicating success or failure
        """
        try:
            logger.info("Starting model rollback")

            # Check if backup exists
            if not Path(MODEL_BACKUP_PATH).exists():
                logger.error("No backup model found for rollback")
                return {"status": "error", "message": "No backup model found"}

            # Replace current model with backup
            shutil.copy2(MODEL_BACKUP_PATH, MODEL_PATH)
            logger.info("Model files restored from backup")

            # Reload the service with the restored model
            self.reload()
            logger.info("Model rollback completed successfully")
            return {"status": "success", "message": "Model rolled back successfully"}

        except Exception as e:
            logger.error(f"Error during model rollback: {e}")
            return {"status": "error", "message": str(e)}

    def _load_model(self):
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _load_features(self):
        try:
            logger.info(f"Loading features from {FEATURES_PATH}")
            with open(FEATURES_PATH, "r") as f:
                features_data = json.load(f)
            logger.info("Features loaded successfully")
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            raise ValueError(f"Failed to load features: {e}")

        if isinstance(features_data, list):
            return features_data + ["zipcode"]
        elif isinstance(features_data, dict) and "features" in features_data:
            return features_data["features"] + ["zipcode"]
        else:
            logger.error("Unrecognized model_features.json format")
            raise ValueError("model_features.json format not recognized")

    def _load_zipcode_data(self):
        try:
            logger.info(f"Loading zipcode data from {ZIPCODE_DATA_PATH}")
            df = pd.read_csv(ZIPCODE_DATA_PATH)
            logger.info(f"Zipcode data loaded: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading zipcode data: {e}")
            raise RuntimeError(f"Failed to load zipcode data: {e}")

    def predict(self, features_dict):
        """
        Make a house price prediction for a single property.

        Takes user-provided house features, merges them with demographic data
        based on ZIP code, and uses the trained model to predict the house price.

        Args:
            features_dict: Dictionary containing house features including:
                          bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                          sqft_above, sqft_basement, zipcode

        Returns:
            Dict containing:
                - predicted_price: Predicted house price in dollars
                - features_used: List of feature names used by the model
                - processing_time_ms: Time taken to process the prediction
                - model_name: Name of the model used for prediction

        Raises:
            RuntimeError: If prediction fails due to data or model issues
        """
        try:
            start_time = time.time()
            logger.debug(f"Starting prediction with features: {features_dict}")

            df = pd.DataFrame([features_dict])
            train_cols = [col for col in SALES_COLUMN_SELECTION if col != TARGET]
            df = df[train_cols]
            df["zipcode"] = df["zipcode"].astype(int)

            logger.debug(f"Merging with zipcode data for zipcode: {features_dict.get('zipcode')}")
            df = df.merge(self.zipcode_df, how="left", on="zipcode")
            df = df[self.features_data]
            df_clean = df.drop(columns="zipcode")

            prediction = self.model.predict(df_clean)[0]
            logger.debug(f"Prediction calculated: {prediction}")

            processing_time = (time.time() - start_time) * 1000

            # Extract final model name if it's a Pipeline
            if hasattr(self.model, 'steps') and len(self.model.steps) > 0:
                model_name = type(self.model.steps[-1][1]).__name__
            else:
                model_name = type(self.model).__name__

            return {
                "predicted_price": float(prediction),
                "features_used": list(df_clean.columns),
                "processing_time_ms": processing_time,
                "model_name": model_name
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_batch(self, features_list):
        """
        Perform batch house price predictions for multiple properties.

        Processes a list of feature dictionaries and returns predictions for each,
        with error handling for individual failed predictions. Failed predictions
        are included in the response with error information rather than failing
        the entire batch.

        Args:
            features_list: List of dictionaries, each containing house features
                         for one property (same format as predict method)

        Returns:
            Dict containing:
                - predictions: List of individual prediction results
                - batch_metadata: Dict with total_predictions count and
                                batch_processing_time_ms

        Raises:
            RuntimeError: If batch processing fails entirely
        """
        try:
            start_time = time.time()
            logger.debug(f"Starting batch prediction for {len(features_list)} items")

            predictions = []
            for i, features_dict in enumerate(features_list):
                try:
                    result = self.predict(features_dict)
                    predictions.append(result)
                except Exception as e:
                    logger.error(f"Error in prediction for item {i}: {e}")
                    # Extract final model name if it's a Pipeline
                    if hasattr(self.model, 'steps') and len(self.model.steps) > 0:
                        model_name = type(self.model.steps[-1][1]).__name__
                    else:
                        model_name = type(self.model).__name__

                    # Add prediction with error
                    predictions.append({
                        "predicted_price": 0.0,
                        "features_used": [],
                        "processing_time_ms": 0.0,
                        "model_name": model_name,
                        "error": str(e)
                    })

            batch_processing_time = (time.time() - start_time) * 1000
            logger.info(f"Batch prediction completed: {len(predictions)} predictions in {batch_processing_time:.2f}ms")

            return {
                "predictions": predictions,
                "batch_metadata": {
                    "total_predictions": len(predictions),
                    "batch_processing_time_ms": batch_processing_time
                }
            }

        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
            raise RuntimeError(f"Batch prediction failed: {e}")

# Global instance
prediction_service = PredictionService()