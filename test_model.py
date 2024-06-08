import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import joblib

def test_model_quality():
    quality_data = pd.read_csv('titanic.csv')
    X_quality = quality_data[['X']]
    y_quality = quality_data['y']
    model = joblib.load('pipeline.pkl')
    y_pred = model.predict(X_quality)
    mse = mean_squared_error(y_quality, y_pred)
    assert mse >= 10, "Model is good enough!"
