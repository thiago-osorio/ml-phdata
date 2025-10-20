import logging
from fastapi import FastAPI, HTTPException
from .config import API_TITLE, API_VERSION
from .models import PredictionMetadata, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse, create_flexible_model
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

        result = model_manager.get_service().predict(feature_dict)
        logger.info(f"Predição realizada com sucesso: {result}")

        metadata = PredictionMetadata(processing_time_ms=result.get("processing_time_ms"))

        return PredictionResponse(
            predicted_price=result["predicted_price"],
            metadata=metadata,
            features_used=result["features_used"]
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

@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """Predição em batch para múltiplos dados"""
    logger.info(f"Recebida requisição de predição em batch para {len(request.predictions)} itens")

    try:
        result = model_manager.get_service().predict_batch(request.predictions)
        logger.info(f"Batch prediction realizada com sucesso: {len(result['predictions'])} predições")

        prediction_responses = []
        for pred in result["predictions"]:
            if "error" in pred:
                metadata = PredictionMetadata(processing_time_ms=pred.get("processing_time_ms", 0.0))
                response = PredictionResponse(
                    predicted_price=pred["predicted_price"],
                    metadata=metadata,
                    features_used=pred["features_used"]
                )
            else:
                metadata = PredictionMetadata(processing_time_ms=pred.get("processing_time_ms"))
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
        logger.error(f"Erro na predição em batch: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro na predição em batch: {str(e)}")