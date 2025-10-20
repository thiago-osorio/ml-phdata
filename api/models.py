from pydantic import BaseModel, create_model, Field
from typing import Union, Optional, List, Dict, Any
from datetime import datetime
import uuid

class PredictionMetadata(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: Optional[float] = None

class PredictionResponse(BaseModel):
    predicted_price: float
    metadata: PredictionMetadata = Field(default_factory=PredictionMetadata)
    features_used: Optional[List[str]] = None

class BatchPredictionRequest(BaseModel):
    predictions: List[Dict[str, Union[int, float]]]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_metadata: Dict[str, Any] = {
        "total_predictions": 0,
        "batch_processing_time_ms": 0.0
    }

def create_flexible_model(feature_names):
    fields = {}
    for feature_name in feature_names + ["zipcode"]:
        fields[feature_name] = Union[int, float]
    return create_model("FlexibleHouseFeatures", **fields)