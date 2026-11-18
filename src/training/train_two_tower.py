"""Week 2 training as a script (same logic as notebooks/week2_two_tower_model.ipynb).

Used by the Airflow weekly retraining DAG. Reads the Week 1 outputs, trains the
two-tower model with early stopping, evaluates it on the test week against the
popularity baseline and exports everything Week 3/4 need to config.MODEL_DIR.

Run:  python -m src.training.train_two_tower
Settings via environment variables: TRAIN_WEEKS, EPOCHS, BATCH_SIZE, LEARNING_RATE.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from src import config
from src.data.features import (apply_standardizer, context_features, encode, fit_standardizer,
                               item_numeric_frame, load_vocabularies, user_numeric_frame)
from src.models.candidate_tower import CandidateTower
from src.models.query_tower import QueryTower
from src.models.two_tower import TwoTowerModel

TRAIN_WEEKS = int(os.environ.get("TRAIN_WEEKS", 12))
VAL_DAYS = 7
EPOCHS = int(os.environ.get("EPOCHS", 10))
PATIENCE = 2
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2048))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
EMBEDDING_DIM, ID_DIM, TEMPERATURE, ID_DROPOUT, TOP_K, SEED = 64, 32, 0.05, 0.1, 12, 42


def trained_item_ids(interactions):
    """Items the candidate tower learned an ID embedding for (same rule as training)."""
    return np.unique(interactions.loc[interactions["days_ago"] > VAL_DAYS, "item_idx"])


def main():
    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    data = config.DATA_PATH

    # ---------------- data
    inter = pd.read_parquet(
        os.path.join(data, "interactions_train.parquet"),
        columns=["user_idx", "item_idx", "t_dat", "month", "day_of_week", "days_ago"],
        filters=[("days_ago", "<=", TRAIN_WEEKS * 7)],
    )
    inter = inter[(inter["user_idx"] > 0) & (inter["item_idx"] > 0)]
    inter = inter.drop_duplicates(["user_idx", "item_idx", "t_dat"]).reset_index(drop=True)
    for c in ["user_idx", "item_idx", "month", "day_of_week", "days_ago"]:
        inter[c] = inter[c].astype("int32")
    train_df = inter[inter["days_ago"] > VAL_DAYS].sample(frac=1.0, random_state=SEED)
    val_df = inter[inter["days_ago"] <= VAL_DAYS]
    test_df = pd.read_parquet(os.path.join(data, "interactions_test.parquet"),
                              columns=["user_idx", "item_idx"])
    test_df = test_df[(test_df["user_idx"] > 0) & (test_df["item_idx"] > 0)]
    print(f"train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    users = pd.read_parquet(os.path.join(data, "user_table.parquet"))
    items = pd.read_parquet(os.path.join(data, "item_table.parquet"))
    vocab = load_vocabularies()
    with open(os.path.join(data, "fallback_top12.json")) as f:
        fallback_top12 = json.load(f)
    n_users, n_items = int(users["user_idx"].max()), int(items["item_idx"].max())

    # ---------------- lookup tables (row = idx, row 0 = unknown)
    u, i = users["user_idx"].to_numpy(), items["item_idx"].to_numpy()
    user_club = np.zeros(n_users + 1, "int32"); user_club[u] = encode(users["club_member_status"], vocab["club_member_status"])
    user_news = np.zeros(n_users + 1, "int32"); user_news[u] = encode(users["fashion_news_frequency"], vocab["fashion_news_frequency"])
    user_frame = user_numeric_frame(users)
    user_mean, user_std = fit_standardizer(user_frame)
    user_num = np.zeros((n_users + 1, len(config.USER_NUM_COLS)), "float32")
    user_num[u] = apply_standardizer(user_frame, user_mean, user_std)

    item_cats = np.zeros((n_items + 1, len(config.ITEM_CAT_COLS)), "int32")
    for col_no, col in enumerate(config.ITEM_CAT_COLS):
        item_cats[i, col_no] = encode(items[col], vocab[col])
    item_frame = item_numeric_frame(items)
    item_mean, item_std = fit_standardizer(item_frame)
    item_num = np.zeros((n_items + 1, len(config.ITEM_NUM_COLS)), "float32")
    item_num[i] = apply_standardizer(item_frame, item_mean, item_std)
    item_article_id = np.full(n_items + 1, "UNKNOWN", dtype=object)
    item_article_id[i] = items["article_id"].to_numpy()

    user_trained = np.zeros(n_users + 1, bool); user_trained[train_df["user_idx"].unique()] = True
    item_trained = np.zeros(n_items + 1, bool); item_trained[trained_item_ids(inter)] = True
    user_trained[0] = item_trained[0] = True
    counts = np.bincount(train_df["item_idx"], minlength=n_items + 1).astype("float64")
    item_log_q = np.log((counts + 1) / (counts.sum() + n_items + 1)).astype("float32")

    def query_features(uidx, month, dow):
        return {"user_idx": np.where(user_trained[uidx], uidx, 0).astype("int32"),
                "club": user_club[uidx], "news": user_news[uidx],
                "month": np.asarray(month, "int32"), "day_of_week": np.asarray(dow, "int32"),
                "user_num": user_num[uidx]}

    def candidate_features(iidx):
        return {"item_idx": np.where(item_trained[iidx], iidx, 0).astype("int32"),
                "cats": item_cats[iidx], "item_num": item_num[iidx]}

    def to_tensors(d):
        return {k: tf.convert_to_tensor(v) for k, v in d.items()}

    def make_dataset(frame, shuffle):
        uidx, iidx = frame["user_idx"].to_numpy(), frame["item_idx"].to_numpy()
        ds = tf.data.Dataset.from_tensor_slices({
            "query": query_features(uidx, frame["month"], frame["day_of_week"]),
            "candidate": candidate_features(iidx),
            "user_key": uidx.astype("int32"), "item_key": iidx.astype("int32"),
            "log_q": item_log_q[iidx],
        })
        if shuffle:
            ds = ds.shuffle(100_000, seed=SEED, reshuffle_each_iteration=True)
        return ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds = make_dataset(train_df, True), make_dataset(val_df, False)

    # ---------------- model + training
    query_tower = QueryTower(n_users, len(vocab["club_member_status"]), len(vocab["fashion_news_frequency"]),
                             EMBEDDING_DIM, ID_DIM, ID_DROPOUT)
    candidate_tower = CandidateTower(n_items, [len(vocab[c]) for c in config.ITEM_CAT_COLS],
                                     EMBEDDING_DIM, ID_DIM, ID_DROPOUT)
    model = TwoTowerModel(query_tower, candidate_tower, TEMPERATURE)
    model.in_batch_loss(next(iter(val_ds)))                # build weights
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            loss = model.in_batch_loss(batch, training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(batch):
        return model.in_batch_loss(batch, training=False)

    history, best = {"train_loss": [], "val_loss": []}, (np.inf, None, 0)
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        t0, tr, va = time.time(), tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
        for b in train_ds:
            tr.update_state(train_step(b))
        for b in val_ds:
            va.update_state(val_step(b))
        history["train_loss"].append(float(tr.result())); history["val_loss"].append(float(va.result()))
        print(f"epoch {epoch}: train {tr.result():.4f} | val {va.result():.4f} | {time.time() - t0:.0f}s")
        if float(va.result()) < best[0] - 1e-4:
            best, bad = (float(va.result()), model.get_weights(), epoch), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stopping")
                break
    model.set_weights(best[1])

    # ---------------- evaluation on the test week
    catalog_idx = np.arange(1, n_items + 1, dtype="int32")
    catalog = tf.concat([candidate_tower(to_tensors(candidate_features(catalog_idx[s:s + 8192])), training=False)
                         for s in range(0, n_items, 8192)], axis=0)
    month, dow = context_features(config.CUTOFF)
    truth = test_df.groupby("user_idx")["item_idx"].apply(set)
    eval_users = truth.index.to_numpy().astype("int32")

    def evaluate(rec_lists):
        tot = {"Recall@12": 0.0, "NDCG@12": 0.0, "MAP@12": 0.0}
        for user, recs in zip(eval_users, rec_lists):
            rel = truth[user]
            hits = np.array([x in rel for x in recs[:TOP_K]], float)
            tot["Recall@12"] += hits.sum() / len(rel)
            tot["NDCG@12"] += (hits / np.log2(np.arange(2, TOP_K + 2))).sum() / \
                (1 / np.log2(np.arange(2, min(len(rel), TOP_K) + 2))).sum()
            tot["MAP@12"] += (np.cumsum(hits) / np.arange(1, TOP_K + 1) * hits).sum() / min(len(rel), TOP_K)
        return {k: v / len(eval_users) for k, v in tot.items()}

    recs = []
    for s in range(0, len(eval_users), 1024):
        b = eval_users[s:s + 1024]
        q = query_tower(to_tensors(query_features(b, np.full(len(b), month), np.full(len(b), dow))), training=False)
        recs.append(tf.math.top_k(tf.matmul(q, catalog, transpose_b=True), k=TOP_K).indices.numpy() + 1)
    model_metrics = evaluate(np.concatenate(recs))
    a2i = dict(zip(items["article_id"], items["item_idx"]))
    baseline = np.array([a2i[a] for a in fallback_top12 if a in a2i])
    baseline_metrics = evaluate([baseline] * len(eval_users))
    print("model   :", {k: round(v, 5) for k, v in model_metrics.items()})
    print("baseline:", {k: round(v, 5) for k, v in baseline_metrics.items()})

    # ---------------- export
    query_tower.save_weights(config.QUERY_WEIGHTS_FILE)
    candidate_tower.save_weights(config.CANDIDATE_WEIGHTS_FILE)
    np.save(config.ITEM_EMBEDDINGS_FILE, catalog.numpy())
    with open(config.ITEM_IDS_FILE, "w") as f:
        json.dump(item_article_id[1:].tolist(), f)
    np.save(config.TRAINED_USERS_FILE, np.flatnonzero(user_trained[1:]) + 1)
    cfg = {
        "embedding_dim": EMBEDDING_DIM, "id_dim": ID_DIM, "temperature": TEMPERATURE,
        "id_dropout": ID_DROPOUT, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "epochs_run": len(history["train_loss"]), "best_epoch": best[2], "best_val_loss": best[0],
        "cutoff": config.CUTOFF, "train_weeks": TRAIN_WEEKS,
        "n_users": n_users, "n_items": n_items,
        "n_club": len(vocab["club_member_status"]), "n_news": len(vocab["fashion_news_frequency"]),
        "item_cat_cols": config.ITEM_CAT_COLS,
        "item_cat_sizes": [len(vocab[c]) for c in config.ITEM_CAT_COLS],
        "user_num_cols": config.USER_NUM_COLS, "user_num_mean": user_mean, "user_num_std": user_std,
        "item_num_cols": config.ITEM_NUM_COLS, "item_num_mean": item_mean, "item_num_std": item_std,
        "test_metrics": {"two_tower_model": model_metrics, "popularity_baseline": baseline_metrics},
        "history": history,
    }
    with open(config.MODEL_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    print("model exported to", config.MODEL_DIR)


if __name__ == "__main__":
    main()
