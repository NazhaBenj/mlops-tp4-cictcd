import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data():
    return pd.read_csv("data/reviews.csv")


def split_data(df, params):
    X = df["text"]
    y = df["label"]

    return train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y
    )