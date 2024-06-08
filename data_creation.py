import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def save_datasets(X_train, X_test, y_train, y_test):
    dir_list = ["train", "test"]
    for ext_dir in dir_list:
        if not os.path.exists(ext_dir):
            os.mkdir(ext_dir)

    X_train.to_csv("train/X_train.csv")
    y_train.to_csv("train/y_train.csv")
    X_test.to_csv("test/X_test.csv")
    y_test.to_csv("test/y_test.csv")


df = pd.read_csv("Task2/titanic.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1
)

save_datasets(X_train, X_test, y_train, y_test)
