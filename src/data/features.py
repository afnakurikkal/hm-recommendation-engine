"""Feature preparation shared by training, embedding export, the Redis loader and the API.

Keeping this in one place guarantees the model sees exactly the same features
at training time and at serving time.
"""
import json
import os

import numpy as np
import pandas as pd

from src import config


def load_vocabularies(data_path=config.DATA_PATH):
    with open(os.path.join(data_path, "vocabularies.json")) as f:
        return json.load(f)


def encode(values, vocab):
    """Category text -> integer code (1..len(vocab)); 0 = unknown."""
    mapping = {value: code for code, value in enumerate(vocab, start=1)}
    return pd.Series(values).map(mapping).fillna(0).astype("int32").to_numpy()


def user_numeric_frame(users):
    """Raw numeric user features (before standardization), columns = config.USER_NUM_COLS."""
    return pd.DataFrame({
        "age": users["age"].astype("float32"),
        "age_missing": users["age_missing"].astype("float32"),
        "log_n_purchases": np.log1p(users["n_purchases"].astype("float32")),
        "log_n_unique_items": np.log1p(users["n_unique_items"].astype("float32")),
        "avg_price": users["avg_price"].astype("float32"),
        "log_days_since_last_purchase": np.log1p(users["days_since_last_purchase"].astype("float32")),
        "is_new_user": users["is_new_user"].astype("float32"),
    })[config.USER_NUM_COLS]


def item_numeric_frame(items):
    """Raw numeric item features (before standardization), columns = config.ITEM_NUM_COLS."""
    return pd.DataFrame({
        "log_purchases_last_7d": np.log1p(items["purchases_last_7d"].astype("float32")),
        "log_purchases_last_28d": np.log1p(items["purchases_last_28d"].astype("float32")),
        "log_purchases_all_time": np.log1p(items["purchases_all_time"].astype("float32")),
        "is_new_item": items["is_new_item"].astype("float32"),
    })[config.ITEM_NUM_COLS]


def fit_standardizer(frame):
    mean = frame.mean()
    std = frame.std().replace(0, 1).fillna(1)
    return {k: float(v) for k, v in mean.items()}, {k: float(v) for k, v in std.items()}


def apply_standardizer(frame, mean, std):
    cols = list(mean.keys())
    m = np.array([mean[c] for c in cols], dtype="float32")
    s = np.array([std[c] for c in cols], dtype="float32")
    return np.nan_to_num((frame[cols].to_numpy(dtype="float32") - m) / s).astype("float32")


def context_features(date=config.CONTEXT_DATE):
    """(month, day_of_week) with Sunday = 1, same as Spark's dayofweek used in Week 1."""
    ts = pd.Timestamp(date)
    return int(ts.month), int((ts.day_of_week + 1) % 7 + 1)
