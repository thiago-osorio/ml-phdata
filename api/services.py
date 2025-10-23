import logging
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import json
import time
from .config import MODEL_PATH, FEATURES_PATH, ZIPCODE_DATA_PATH, SALES_DATA_PATH, SALES_COLUMN_SELECTION, TARGET

logger = logging.getLogger(__name__)

class TrainService:
    def __init__(self):
        logger.info("Inicializando TrainService")
        self.data = self._load_data()
        
        logger.info("TrainService inicializado com sucesso")
    
    def _load_data(self) -> Dict[str, Any]:
        data = pd.read_csv(SALES_DATA_PATH, usecols=SALES_COLUMN_SELECTION, dtype={'zipcode': str})
        demographics = pd.read_csv(ZIPCODE_DATA_PATH,dtype={'zipcode': str})

        merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
        y = merged_data.pop('price')
        x = merged_data
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
        
        return {
            "x_train": x_train, 
            "y_train": y_train, 
            "x_test": x_test, 
            "y_test": y_test
            }

    def _train_model(self):
        logger.info("Iniciando treinamento do modelo")
        
        pipe = make_pipeline(RobustScaler(), RandomForestRegressor())
        pipe.fit(self.data["x_train"], self.data["y_train"])
        pred = pipe.predict(self.data["x_test"])
        mape = mean_absolute_percentage_error(y_true=self.data["y_test"], y_pred=pred)
        logger.info(f"MAPE {round(mape, 4)}")
        pickle.dump(pipe, open(MODEL_PATH, 'wb'))
        
        logger.info("Modelo treinado com sucesso")

class PredictionService:
    def __init__(self, model=None, features_data=None, zipcode_df=None):
        logger.info("Initializing PredictionService")

        if model is not None and features_data is not None and zipcode_df is not None:
            self.model = model
            self.features_data = features_data if "zipcode" in features_data else features_data + ["zipcode"]
            self.zipcode_df = zipcode_df
        else:
            self.model = self._load_model()
            self.features_data = self._load_features()
            self.zipcode_df = self._load_zipcode_data()

        logger.info("PredictionService initialized successfully")

    def reload(self):
        """Reload the model, features, and zipcode data."""
        try:
            logger.info("Reloading PredictionService components")
            self.model = self._load_model()
            self.features_data = self._load_features()
            self.zipcode_df = self._load_zipcode_data()
            logger.info("PredictionService reloaded successfully")
            return {"status": "success", "message": "Model reloaded successfully"}
        except Exception as e:
            logger.error(f"Error reloading PredictionService: {e}")
            return {"status": "error", "message": str(e)}

    def retrain(self):
        """Retrain the model using TrainService."""
        try:
            logger.info("Starting model retraining")
            train_service = TrainService()
            train_service._train_model()
            # Reload after training
            self.reload()
            logger.info("Model retrained and reloaded successfully")
            return {"status": "success", "message": "Model retrained successfully"}
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return {"status": "error", "message": str(e)}

    def _load_model(self):
        try:
            logger.info(f"Carregando modelo de {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info("Modelo carregado com sucesso")
            return model
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise RuntimeError(f"Falha ao carregar modelo: {e}")

    def _load_features(self):
        try:
            logger.info(f"Carregando features de {FEATURES_PATH}")
            with open(FEATURES_PATH, "r") as f:
                features_data = json.load(f)
            logger.info("Features carregadas com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar features: {e}")
            raise ValueError(f"Falha ao carregar features: {e}")

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
            raise RuntimeError(f"Falha ao carregar dados de zipcode: {e}")

    def predict(self, features_dict):
        try:
            start_time = time.time()
            logger.debug(f"Iniciando predição com features: {features_dict}")

            df = pd.DataFrame([features_dict])
            train_cols = [col for col in SALES_COLUMN_SELECTION if col != TARGET]
            df = df[train_cols]
            df["zipcode"] = df["zipcode"].astype(int)

            logger.debug(f"Fazendo merge com dados de zipcode para zipcode: {features_dict.get('zipcode')}")
            df = df.merge(self.zipcode_df, how="left", on="zipcode")
            df = df[self.features_data]
            df_clean = df.drop(columns="zipcode")

            prediction = self.model.predict(df_clean)[0]
            logger.debug(f"Predição calculada: {prediction}")

            processing_time = (time.time() - start_time) * 1000

            # Extrair nome do modelo final se for um Pipeline
            if hasattr(self.model, 'steps') and len(self.model.steps) > 0:
                model_name = type(self.model.steps[-1][1]).__name__
            else:
                model_name = type(self.model).__name__

            return {
                "predicted_price": float(prediction),
                "features_used": list(df_clean.columns),
                "processing_time_ms": processing_time,
                "model_name": model_name
            }

        except Exception as e:
            logger.error(f"Erro durante predição: {e}")
            raise RuntimeError(f"Falha na predição: {e}")

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
                    # Extrair nome do modelo final se for um Pipeline
                    if hasattr(self.model, 'steps') and len(self.model.steps) > 0:
                        model_name = type(self.model.steps[-1][1]).__name__
                    else:
                        model_name = type(self.model).__name__

                    # Adiciona predição com erro
                    predictions.append({
                        "predicted_price": 0.0,
                        "features_used": [],
                        "processing_time_ms": 0.0,
                        "model_name": model_name,
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
            raise RuntimeError(f"Falha na predição em batch: {e}")

# Global instance
prediction_service = PredictionService()