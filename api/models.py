from pydantic import BaseModel, create_model
from typing import Union

class PredictionResponse(BaseModel):
    predicted_price: float

def create_flexible_model(feature_names):
    fields = {}
    for feature_name in feature_names + ["zipcode"]:
        fields[feature_name] = Union[int, float]
    return create_model("FlexibleHouseFeatures", **fields)