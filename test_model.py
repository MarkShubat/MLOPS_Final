import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

def test_model_quality():
    quality_data = pd.read_csv('titanic.csv')
    X_quality = quality_data.drop('Survived', axis=1)
    y_quality = quality_data['Survived']
    model = joblib.load('pipeline.pkl')
    y_pred = model.predict(X_quality)
    mse = mean_squared_error(y_quality, y_pred)
    assert mse >= 10, "Model is good enough!"
