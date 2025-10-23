import logging
from typing import List, Dict, Union
from fastapi import FastAPI, HTTPException
from .config import API_TITLE, API_VERSION
from .models import PredictionRequest, PredictionMetadata, PredictionResponse, BatchPredictionResponse
from .services import prediction_service

logger = logging.getLogger(__name__)

app = FastAPI(title=API_TITLE, version=API_VERSION)

@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: PredictionRequest):
    logger.info("Recebida requisição de predição")

    try:
        feature_dict = features.model_dump(exclude_none=True)
        logger.debug(f"Features recebidas: {feature_dict}")

        result = prediction_service.predict(feature_dict)
        logger.info(f"Predição realizada com sucesso: {result}")

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
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro na predição: {str(e)}")

@app.post("/reload-model")
def reload_model():
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
    logger.info("Received request to retrain model")

    try:
        result = prediction_service.retrain()
        logger.info(f"Retrain result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retrain error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(features_list: List[PredictionRequest]):
    """Predição em batch para múltiplos dados"""
    logger.info(f"Recebida requisição de predição em batch para {len(features_list)} itens")

    try:
        features_dicts = [feature.model_dump(exclude_none=True) for feature in features_list]
        result = prediction_service.predict_batch(features_dicts)
        logger.info(f"Batch prediction realizada com sucesso: {len(result['predictions'])} predições")

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
        logger.error(f"Erro na predição em batch: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro na predição em batch: {str(e)}")