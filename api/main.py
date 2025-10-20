import logging
from fastapi import FastAPI, HTTPException
from .config import API_TITLE, API_VERSION
from .models import PredictionResponse, create_flexible_model
from .services import PredictionService

logger = logging.getLogger(__name__)

app = FastAPI(title=API_TITLE, version=API_VERSION)

prediction_service = PredictionService()
FlexibleHouseFeatures = create_flexible_model(prediction_service.features_data)

@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: FlexibleHouseFeatures):
    """Predição com enhancement automático de zipcode"""
    logger.info("Recebida requisição de predição")

    try:
        feature_dict = features.model_dump(exclude_none=True)
        logger.debug(f"Features recebidas: {feature_dict}")

        prediction = prediction_service.predict(feature_dict)
        logger.info(f"Predição realizada com sucesso: {prediction}")

        return PredictionResponse(
            predicted_price=float(prediction)
        )

    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro na predição: {str(e)}")