"""Daily embedding refresh (used by the Airflow daily DAG).

Item popularity features change every day, so the item vectors are recomputed
with the already-trained candidate tower, without retraining the model.

Run:  python -m src.serving.export_embeddings
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src import config
from src.data.features import apply_standardizer, encode, item_numeric_frame, load_vocabularies
from src.models.loader import load_candidate_tower, load_model_config

VAL_DAYS = 7


def main():
    cfg = load_model_config()
    vocab = load_vocabularies()
    items = pd.read_parquet(os.path.join(config.DATA_PATH, "item_table.parquet")).sort_values("item_idx")
    n_items = cfg["n_items"]

    # items the tower has a trained ID embedding for (same rule as training)
    inter = pd.read_parquet(
        os.path.join(config.DATA_PATH, "interactions_train.parquet"), columns=["item_idx", "days_ago"],
        filters=[("days_ago", "<=", cfg["train_weeks"] * 7), ("days_ago", ">", VAL_DAYS)],
    )
    item_trained = np.zeros(n_items + 1, bool)
    item_trained[inter["item_idx"].unique()] = True
    item_trained[0] = True

    i = items["item_idx"].to_numpy()
    cats = np.zeros((n_items + 1, len(config.ITEM_CAT_COLS)), "int32")
    for col_no, col in enumerate(config.ITEM_CAT_COLS):
        cats[i, col_no] = encode(items[col], vocab[col])
    num = np.zeros((n_items + 1, len(config.ITEM_NUM_COLS)), "float32")
    num[i] = apply_standardizer(item_numeric_frame(items), cfg["item_num_mean"], cfg["item_num_std"])

    tower = load_candidate_tower(cfg)
    idx = np.arange(1, n_items + 1, dtype="int32")
    parts = []
    for s in range(0, n_items, 8192):
        b = idx[s:s + 8192]
        parts.append(tower({"item_idx": tf.constant(np.where(item_trained[b], b, 0).astype("int32")),
                            "cats": tf.constant(cats[b]), "item_num": tf.constant(num[b])},
                           training=False).numpy())
    embeddings = np.concatenate(parts).astype("float32")
    np.save(config.ITEM_EMBEDDINGS_FILE, embeddings)
    print("item embeddings refreshed:", embeddings.shape, "->", config.ITEM_EMBEDDINGS_FILE)


if __name__ == "__main__":
    main()
