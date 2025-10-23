"""
Data models for the House Price Prediction API.

This module contains Pydantic models that define the structure of requests and responses
for the house price prediction service. These models handle data validation, serialization,
and provide clear API documentation through type hints.
"""

from pydantic import BaseModel, create_model, Field, ConfigDict
from typing import Union, Optional, List, Dict, Any
from datetime import datetime
import uuid


class PredictionRequest(BaseModel):
    """
    Model for house price prediction requests.

    This model defines the required and optional features that clients must provide
    when requesting a house price prediction. All features represent characteristics
    of a house that influence its market value.

    Attributes:
        bedrooms: Number of bedrooms in the house
        bathrooms: Number of bathrooms (can include partial bathrooms as decimals)
        sqft_living: Square footage of the living space
        sqft_lot: Square footage of the lot/property
        floors: Number of floors (can be fractional for split-level homes)
        sqft_above: Square footage above ground level
        sqft_basement: Square footage of basement space
        zipcode: ZIP code where the house is located (used for demographic data)
    """
    model_config = ConfigDict(extra="allow")
    bedrooms: Union[int, float]
    bathrooms: Union[int, float]
    sqft_living: Union[int, float]
    sqft_lot: Union[int, float]
    floors: Union[int, float]
    sqft_above: Union[int, float]
    sqft_basement: Union[int, float]
    zipcode: Union[int, float]

class PredictionMetadata(BaseModel):
    """
    Metadata information for prediction responses.

    This model contains auxiliary information about the prediction process,
    including timing data, model information, and unique identifiers for
    tracking and debugging purposes.

    Attributes:
        prediction_id: Unique 8-character identifier for this prediction
        timestamp: ISO format timestamp when the prediction was made
        processing_time_ms: Time taken to process the prediction in milliseconds
        model_name: Name of the machine learning model used for prediction
    """
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: Optional[float] = None
    model_name: Optional[str] = None

class PredictionResponse(BaseModel):
    """
    Response model for individual house price predictions.

    This model represents the complete response from a price prediction request,
    including the predicted price, metadata about the prediction process, and
    information about which features were used in the model.

    Attributes:
        predicted_price: The predicted house price in dollars
        metadata: Metadata about the prediction process and timing
        features_used: List of feature names that were used by the model
    """
    predicted_price: float
    metadata: PredictionMetadata = Field(default_factory=PredictionMetadata)
    features_used: Optional[List[str]] = None

class BatchPredictionResponse(BaseModel):
    """
    Response model for batch house price predictions.

    This model represents the response from a batch prediction request,
    containing multiple individual predictions along with batch-level
    metadata and performance statistics.

    Attributes:
        predictions: List of individual prediction responses
        batch_metadata: Dictionary containing batch processing statistics including
                       total_predictions count and batch_processing_time_ms
    """
    predictions: List[PredictionResponse]
    batch_metadata: Dict[str, Any] = {
        "total_predictions": 0,
        "batch_processing_time_ms": 0.0
    }

class RequiredFeaturesResponse(BaseModel):
    """
    Response model for the required features endpoint.

    This model provides information about which features clients must provide
    when making prediction requests, excluding features that are automatically
    joined from demographic datasets.

    Attributes:
        required_features: List of feature names that must be provided by the user
        description: Human-readable description of what these features represent
    """
    required_features: List[str]
    description: str = "Features that must be provided by the user for prediction (excludes zipcode demographic features)"