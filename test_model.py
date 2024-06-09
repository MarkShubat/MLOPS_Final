import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def test_model_quality():
    quality_data = pd.read_csv('titanic.csv')
    print(quality_data.info())
    X = quality_data.drop('Survived', axis=1)
    y = quality_data['Survived']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=1) 
    pipeline = joblib.load('pipeline.pkl')
    preds = pipeline.predict(X_valid)
    a = accuracy_score(y_valid, preds)
    assert a >= 0.75, "Model is bad!"
