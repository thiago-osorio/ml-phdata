from pydantic import BaseModel, create_model
from typing import Optional, Union

class PredictionResponse(BaseModel):
    predicted_price: float

def create_flexible_model(feature_names):
    fields = {}
    for feature_name in feature_names + ["zipcode"]:
        fields[feature_name] = (Optional[Union[str, int, float, bool]], None)
    return create_model("FlexibleHouseFeatures", **fields)