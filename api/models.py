from pydantic import BaseModel, create_model, Field, ConfigDict
from typing import Union, Optional, List, Dict, Any
from datetime import datetime
import uuid

class PredictionRequest(BaseModel):
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
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: Optional[float] = None
    model_name: Optional[str] = None

class PredictionResponse(BaseModel):
    predicted_price: float
    metadata: PredictionMetadata = Field(default_factory=PredictionMetadata)
    features_used: Optional[List[str]] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_metadata: Dict[str, Any] = {
        "total_predictions": 0,
        "batch_processing_time_ms": 0.0
    }

class RequiredFeaturesResponse(BaseModel):
    required_features: List[str]
    description: str = "Features that must be provided by the user for prediction (excludes zipcode demographic features)"