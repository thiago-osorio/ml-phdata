import pandas as pd
import requests
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from create_model import load_data, SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION

x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# KNN
## Train
print("Starting KNN Model\n\n")
pipe = make_pipeline(RobustScaler(), KNeighborsRegressor()).fit(x_train, y_train)
pred = pipe.predict(x_train)
r2 = r2_score(y_true=y_train, y_pred=pred)
mae = mean_absolute_error(y_true=y_train, y_pred=pred)
mape = mean_absolute_percentage_error(y_true=y_train, y_pred=pred)
print(f"""
    Train Summary KNN:
    R2    - {round(r2, 4)}
    MAE   - {round(mae, 4)}
    MAPE  - {round(mape, 4)}\n\n
    """)

## Test
pred = pipe.predict(x_test)
r2 = r2_score(y_true=y_test, y_pred=pred)
mae = mean_absolute_error(y_true=y_test, y_pred=pred)
mape = mean_absolute_percentage_error(y_true=y_test, y_pred=pred)
print(f"""
    Test Summary KNN:
    R2    - {round(r2, 4)}
    MAE   - {round(mae, 4)}
    MAPE  - {round(mape, 4)}\n\n
    """)
print("Finished KNN Model\n\n")

# RandomForest
## Train
print("Starting RandomForest Model\n\n")
pipe = make_pipeline(RobustScaler(), RandomForestRegressor()).fit(x_train, y_train)
pred = pipe.predict(x_train)
r2 = r2_score(y_true=y_train, y_pred=pred)
mae = mean_absolute_error(y_true=y_train, y_pred=pred)
mape = mean_absolute_percentage_error(y_true=y_train, y_pred=pred)
print(f"""
    Train Summary RandomForest:
    R2    - {round(r2, 4)}
    MAE   - {round(mae, 4)}
    MAPE  - {round(mape, 4)}\n\n
    """)

pred = pipe.predict(x_test)
r2 = r2_score(y_true=y_test, y_pred=pred)
mae = mean_absolute_error(y_true=y_test, y_pred=pred)
mape = mean_absolute_percentage_error(y_true=y_test, y_pred=pred)
print(f"""
    Test Summary RandomForest:
    R2    - {round(r2, 4)}
    MAE   - {round(mae, 4)}
    MAPE  - {round(mape, 4)}\n\n
    """)
print("Finished RandomForest Model\n\n")

# API Test
print("Starting API Test\n\n")
df = pd.read_csv("data/future_unseen_examples.csv")

print("Showing Model Required features\n\n")
response = requests.get("http://localhost:8000/features/required", json=df.iloc[0].to_dict())
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

print("Predicting first row using all features\n\n")
response = requests.post("http://localhost:8000/predict", json=df.iloc[0].to_dict())
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

print("Predicting first row using required features\n\n")
response = requests.post("http://localhost:8000/predict", json=df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','sqft_above', 'sqft_basement', 'zipcode']].iloc[0].to_dict())
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

print("Batch Prediction using all features\n\n")
response = requests.post("http://localhost:8000/predict-batch", json=df.to_dict(orient="records"))
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

print("Batch Prediction using required features\n\n")
response = requests.post("http://localhost:8000/predict-batch", json=df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','sqft_above', 'sqft_basement', 'zipcode']].to_dict(orient="records"))
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

print("Training RandomForest Model")
response = requests.post("http://localhost:8000/retrain-model")
print(response)

print("Predicting first row using all features and RandomForest\n\n")
response = requests.post("http://localhost:8000/predict", json=df.iloc[0].to_dict())
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

print("Reloading RandomForest Model\n\n")
response = requests.post("http://localhost:8000/reload-model")
print(f"Response Text: {response.text}")

print("RollingBack to KNN Model\n\n")
response = requests.post("http://localhost:8000/rollback-model")
print(f"Response Text: {response.text}")

print("Predicting first row using all features\n\n")
response = requests.post("http://localhost:8000/predict", json=df.iloc[0].to_dict())
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")