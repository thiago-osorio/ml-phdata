"""
FastAPI application for house price prediction.

This module defines the REST API endpoints for the house price prediction service.
It provides endpoints for single and batch predictions, model management operations,
and feature information retrieval.

The API uses FastAPI for automatic documentation generation, request/response validation,
and high-performance async handling. All endpoints include comprehensive error handling
and logging for monitoring and debugging.
"""

import logging
from typing import List, Dict, Union
from fastapi import FastAPI, HTTPException
from .config import API_TITLE, API_VERSION, SALES_COLUMN_SELECTION, TARGET
from .models import PredictionRequest, PredictionMetadata, PredictionResponse, BatchPredictionResponse, RequiredFeaturesResponse
from .services import prediction_service

logger = logging.getLogger(__name__)

app = FastAPI(title=API_TITLE, version=API_VERSION)

@app.get("/features/required", response_model=RequiredFeaturesResponse)
def get_required_features():
    """
    Get the list of features required for house price predictions.

    This endpoint returns information about which features clients must provide
    when making prediction requests. It excludes features that are automatically
    joined from demographic datasets based on ZIP code.

    Returns:
        RequiredFeaturesResponse: Object containing:
            - required_features: List of feature names that must be provided
            - description: Human-readable explanation of the features

    Raises:
        HTTPException: 500 if there's an error retrieving feature information

    Example:
        GET /features/required
        Response: {
            "required_features": ["bedrooms", "bathrooms", "sqft_living", ...],
            "description": "Features that must be provided by the user..."
        }
    """
    logger.info("Received request for required features")

    try:
        # Features that user must provide (from SALES_COLUMN_SELECTION, excluding price which is the target)
        required_features = [col for col in SALES_COLUMN_SELECTION if col != TARGET]

        logger.info(f"Returning {len(required_features)} required features")

        return RequiredFeaturesResponse(
            required_features=required_features
        )

    except Exception as e:
        logger.error(f"Error getting required features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting required features: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: PredictionRequest):
    """
    Predict the price of a house based on its features.

    This endpoint takes house characteristics and returns a predicted market price
    using a trained machine learning model. The prediction automatically incorporates
    demographic data based on the provided ZIP code.

    Args:
        features: PredictionRequest object containing house features including:
                 bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                 sqft_above, sqft_basement, zipcode

    Returns:
        PredictionResponse: Object containing:
            - predicted_price: Estimated house price in dollars
            - metadata: Processing information (ID, timestamp, timing, model name)
            - features_used: List of all features used by the model

    Raises:
        HTTPException: 400 if prediction fails due to invalid input or model errors

    Example:
        POST /predict
        Body: {
            "bedrooms": 3,
            "bathrooms": 2.5,
            "sqft_living": 2000,
            "sqft_lot": 8000,
            "floors": 2,
            "sqft_above": 1500,
            "sqft_basement": 500,
            "zipcode": 98105
        }
        Response: {
            "predicted_price": 675000.50,
            "metadata": {...},
            "features_used": [...]
        }
    """
    logger.info("Received prediction request")

    try:
        feature_dict = features.model_dump(exclude_none=True)
        logger.debug(f"Received features: {feature_dict}")

        result = prediction_service.predict(feature_dict)
        logger.info(f"Prediction completed successfully: {result}")

        metadata = PredictionMetadata(
            processing_time_ms=result.get("processing_time_ms"),
            model_name=result.get("model_name")
        )

        return PredictionResponse(
            predicted_price=result["predicted_price"],
            metadata=metadata,
            features_used=result["features_used"]
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/reload-model")
def reload_model():
    """
    Reload the machine learning model and associated data from disk.

    This endpoint refreshes the prediction service by reloading the model,
    feature definitions, and demographic data from their respective files.
    Useful after external model updates or configuration changes.

    Returns:
        Dict: Status object with 'status' and 'message' fields indicating
              success or failure of the reload operation

    Raises:
        HTTPException: 500 if model reload fails

    Example:
        POST /reload-model
        Response: {
            "status": "success",
            "message": "Model reloaded successfully"
        }
    """
    logger.info("Received request to reload model")

    try:
        result = prediction_service.reload()
        logger.info(f"Reload result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reload error: {str(e)}")

@app.post("/retrain-model")
def retrain_model():
    """
    Retrain the machine learning model with fresh data.

    This endpoint triggers a complete model retraining process using the latest
    available data. It automatically creates a backup of the current model before
    training, then replaces the active model with the newly trained version.

    Returns:
        Dict: Status object with 'status' and 'message' fields indicating
              success or failure, including backup creation status

    Raises:
        HTTPException: 500 if model retraining fails

    Example:
        POST /retrain-model
        Response: {
            "status": "success",
            "message": "Model retrained successfully (backup created)"
        }

    Note:
        This operation may take several minutes to complete depending on
        data size and model complexity.
    """
    logger.info("Received request to retrain model")

    try:
        result = prediction_service.retrain()
        logger.info(f"Retrain result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retrain error: {str(e)}")

@app.post("/rollback-model")
def rollback_model():
    """
    Rollback to the previous version of the machine learning model.

    This endpoint restores the model from the backup created during the last
    retraining operation. This provides a quick recovery mechanism if the
    newly trained model performs poorly or has issues.

    Returns:
        Dict: Status object with 'status' and 'message' fields indicating
              success or failure of the rollback operation

    Raises:
        HTTPException: 500 if rollback fails (e.g., no backup exists)

    Example:
        POST /rollback-model
        Response: {
            "status": "success",
            "message": "Model rolled back successfully"
        }

    Note:
        Rollback is only possible if a backup was created during a previous
        retrain operation. If no backup exists, the request will fail.
    """
    logger.info("Received request to rollback model")

    try:
        result = prediction_service.rollback()
        logger.info(f"Rollback result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error rolling back model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rollback error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(features_list: List[PredictionRequest]):
    """
    Predict prices for multiple houses in a single batch request.

    This endpoint processes a list of house feature sets and returns predictions
    for all of them. It provides better performance for multiple predictions
    compared to individual requests, with robust error handling that allows
    partial success (failed individual predictions don't fail the entire batch).

    Args:
        features_list: List of PredictionRequest objects, each containing
                      house features for one property

    Returns:
        BatchPredictionResponse: Object containing:
            - predictions: List of individual prediction results
            - batch_metadata: Performance statistics including total count
                            and processing time

    Raises:
        HTTPException: 400 if batch processing fails entirely

    Example:
        POST /predict-batch
        Body: [
            {
                "bedrooms": 3,
                "bathrooms": 2,
                "sqft_living": 1800,
                "zipcode": 98105,
                ...
            },
            {
                "bedrooms": 4,
                "bathrooms": 3,
                "sqft_living": 2500,
                "zipcode": 98115,
                ...
            }
        ]
        Response: {
            "predictions": [...],
            "batch_metadata": {
                "total_predictions": 2,
                "batch_processing_time_ms": 45.2
            }
        }

    Note:
        Individual prediction failures are included in the response with
        error information rather than failing the entire batch.
    """
    logger.info(f"Received batch prediction request for {len(features_list)} items")

    try:
        features_dicts = [feature.model_dump(exclude_none=True) for feature in features_list]
        result = prediction_service.predict_batch(features_dicts)
        logger.info(f"Batch prediction completed successfully: {len(result['predictions'])} predictions")

        prediction_responses = []
        for pred in result["predictions"]:
            if "error" in pred:
                metadata = PredictionMetadata(
                    processing_time_ms=pred.get("processing_time_ms", 0.0),
                    model_name=pred.get("model_name")
                )
                response = PredictionResponse(
                    predicted_price=pred["predicted_price"],
                    metadata=metadata,
                    features_used=pred["features_used"]
                )
            else:
                metadata = PredictionMetadata(
                    processing_time_ms=pred.get("processing_time_ms"),
                    model_name=pred.get("model_name")
                )
                response = PredictionResponse(
                    predicted_price=pred["predicted_price"],
                    metadata=metadata,
                    features_used=pred["features_used"]
                )
            prediction_responses.append(response)

        return BatchPredictionResponse(
            predictions=prediction_responses,
            batch_metadata=result["batch_metadata"]
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")