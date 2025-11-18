import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train():
    df = pd.read_csv("data/transactions.csv")
    X = df.drop("fraud", axis=1)
    y = df["fraud"]

    model = RandomForestClassifier()
    model.fit(X, y)
    return model
