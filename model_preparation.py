from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from model_preprocessing import preprocessor
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import os
import pandas as pd


model = RandomForestClassifier(n_estimators=100, random_state=0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])
df = pd.read_csv("Task2/titanic.csv")

X = pd.read_csv("train/X_train.csv")
y = pd.read_csv("train/y_train.csv")["Survived"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=1
)
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)
print('Model accuracy score:', accuracy_score(y_valid, preds))

with open('pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)
