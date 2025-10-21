import os
import pickle
import json
import pandas as pd
from threading import Lock
from .services import PredictionService, TrainService
from .config import MODEL_PATH, FEATURES_PATH, ZIPCODE_DATA_PATH

class ModelManager:
    def __init__(self):
        self.lock = Lock()
        self.prediction_service = None
        self.load_model()
        self.train_service = TrainService()

    def load_model(self):
        with self.lock:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)

            with open(FEATURES_PATH, 'r') as f:
                features = json.load(f)

            zipcode_df = pd.read_csv(ZIPCODE_DATA_PATH)

            self.prediction_service = PredictionService(
                model=model,
                features_data=features,
                zipcode_df=zipcode_df
            )

    def get_service(self):
        with self.lock:
            return self.prediction_service

    def reload_model(self):
        try:
            self.load_model()
            return {"status": "success", "message": "Model reloaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def retrain_model(self):
        try:
            self.train_service._train_model()
            return {"status": "success", "message": "Model retrained successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

model_manager = ModelManager()