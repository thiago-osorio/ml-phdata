import logging
from fastapi import FastAPI, HTTPException
from .config import API_TITLE, API_VERSION
from .models import PredictionResponse, create_flexible_model
from .model_manager import model_manager

logger = logging.getLogger(__name__)

app = FastAPI(title=API_TITLE, version=API_VERSION)

FlexibleHouseFeatures = create_flexible_model(model_manager.get_service().features_data)

@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: FlexibleHouseFeatures):
    logger.info("Recebida requisição de predição")

    try:
        feature_dict = features.model_dump(exclude_none=True)
        logger.debug(f"Features recebidas: {feature_dict}")

        prediction = model_manager.get_service().predict(feature_dict)
        logger.info(f"Predição realizada com sucesso: {prediction}")

        return PredictionResponse(
            predicted_price=float(prediction)
        )

    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro na predição: {str(e)}")

@app.post("/reload-model")
def reload_model():
    logger.info("Recebida requisição para recarregar modelo")

    try:
        result = model_manager.reload_model()
        logger.info(f"Resultado do reload: {result}")
        return result
    except Exception as e:
        logger.error(f"Erro no reload do modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no reload: {str(e)}")