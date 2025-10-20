import logging
import pandas as pd
import pickle
import json
import time
from .config import MODEL_PATH, FEATURES_PATH, ZIPCODE_DATA_PATH
from .exceptions import ModelLoadError, FeaturesLoadError, DataFrameLoadError, PredictionError

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model=None, features_data=None, zipcode_df=None):
        logger.info("Inicializando PredictionService")

        if model is not None and features_data is not None and zipcode_df is not None:
            self.model = model
            self.features_data = features_data if "zipcode" in features_data else features_data + ["zipcode"]
            self.zipcode_df = zipcode_df
        else:
            self.model = self._load_model()
            self.features_data = self._load_features()
            self.zipcode_df = self._load_zipcode_data()

        logger.info("PredictionService inicializado com sucesso")

    def _load_model(self):
        try:
            logger.info(f"Carregando modelo de {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info("Modelo carregado com sucesso")
            return model
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise ModelLoadError(f"Falha ao carregar modelo: {e}")

    def _load_features(self):
        try:
            logger.info(f"Carregando features de {FEATURES_PATH}")
            with open(FEATURES_PATH, "r") as f:
                features_data = json.load(f)
            logger.info("Features carregadas com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar features: {e}")
            raise FeaturesLoadError(f"Falha ao carregar features: {e}")

        if isinstance(features_data, list):
            return features_data + ["zipcode"]
        elif isinstance(features_data, dict) and "features" in features_data:
            return features_data["features"] + ["zipcode"]
        else:
            logger.error("Formato de model_features.json não reconhecido")
            raise ValueError("model_features.json format not recognized")

    def _load_zipcode_data(self):
        try:
            logger.info(f"Carregando dados de zipcode de {ZIPCODE_DATA_PATH}")
            df = pd.read_csv(ZIPCODE_DATA_PATH)
            logger.info(f"Dados de zipcode carregados: {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar dados de zipcode: {e}")
            raise DataFrameLoadError(f"Falha ao carregar dados de zipcode: {e}")

    def predict(self, features_dict):
        try:
            start_time = time.time()
            logger.debug(f"Iniciando predição com features: {features_dict}")

            df = pd.DataFrame([features_dict])
            df["zipcode"] = df["zipcode"].astype(int)

            logger.debug(f"Fazendo merge com dados de zipcode para zipcode: {features_dict.get('zipcode')}")
            df = df.merge(self.zipcode_df, how="left", on="zipcode")
            df = df[self.features_data]
            df_clean = df.drop(columns="zipcode")

            prediction = self.model.predict(df_clean)[0]
            logger.debug(f"Predição calculada: {prediction}")

            processing_time = (time.time() - start_time) * 1000

            return {
                "predicted_price": float(prediction),
                "features_used": list(df_clean.columns),
                "processing_time_ms": processing_time
            }

        except Exception as e:
            logger.error(f"Erro durante predição: {e}")
            raise PredictionError(f"Falha na predição: {e}")

    def predict_batch(self, features_list):
        """Realiza predições em batch para uma lista de features"""
        try:
            start_time = time.time()
            logger.debug(f"Iniciando predição em batch para {len(features_list)} itens")

            predictions = []
            for i, features_dict in enumerate(features_list):
                try:
                    result = self.predict(features_dict)
                    predictions.append(result)
                except Exception as e:
                    logger.error(f"Erro na predição do item {i}: {e}")
                    # Adiciona predição com erro
                    predictions.append({
                        "predicted_price": 0.0,
                        "features_used": [],
                        "processing_time_ms": 0.0,
                        "error": str(e)
                    })

            batch_processing_time = (time.time() - start_time) * 1000
            logger.info(f"Batch prediction concluída: {len(predictions)} predições em {batch_processing_time:.2f}ms")

            return {
                "predictions": predictions,
                "batch_metadata": {
                    "total_predictions": len(predictions),
                    "batch_processing_time_ms": batch_processing_time
                }
            }

        except Exception as e:
            logger.error(f"Erro durante predição em batch: {e}")
            raise PredictionError(f"Falha na predição em batch: {e}")