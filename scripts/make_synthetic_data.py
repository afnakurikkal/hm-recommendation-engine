"""Tiny synthetic dataset in the exact Week 1 output format.

Used by the automated tests (tests/test_pipeline.py) so the full pipeline -
training, ANN index, feature store and API - can be checked in about a minute
on any machine, without downloading the 30 GB H&M dataset. Customers have a
favourite product category, so the model has a real pattern to learn.

    python scripts/make_synthetic_data.py --out data/synthetic
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCT_TYPES = ["Trousers", "Dress", "Sweater", "T-shirt", "Jacket", "Skirt", "Shorts", "Blouse"]
COLOURS = ["Black", "White", "Blue", "Red", "Beige", "Grey", "Pink", "Green"]
DEPARTMENTS = ["Jersey Basic", "Trousers", "Dresses", "Knitwear", "Outdoor", "Blouse"]
GARMENTS = ["Jersey Basic", "Trousers", "Dresses Ladies", "Knitwear", "Outdoor", "Blouses"]


def make(out_dir, n_users=400, n_items=240, n_weeks=14, cutoff="2020-09-16", seed=7):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cutoff_ts = pd.Timestamp(cutoff)

    # ---------------- articles
    item_type = rng.integers(0, len(PRODUCT_TYPES), n_items)
    article_ids = [f"0{100000000 + i * 1001:09d}"[:10] for i in range(n_items)]
    items = pd.DataFrame({
        "article_id": article_ids,
        "item_idx": np.arange(1, n_items + 1, dtype="int32"),
        "product_type_name": [PRODUCT_TYPES[t] for t in item_type],
        "colour_group_name": rng.choice(COLOURS, n_items),
        "department_name": rng.choice(DEPARTMENTS, n_items),
        "garment_group_name": rng.choice(GARMENTS, n_items),
    })

    # ---------------- customers (each has a favourite product type); last 10% never bought anything
    favourite = rng.integers(0, len(PRODUCT_TYPES), n_users)
    customer_ids = [f"{i:064x}" for i in rng.integers(1, 2**62, n_users)]
    n_active = int(n_users * 0.9)

    # ---------------- transactions: 85% from the favourite type
    rows = []
    for u in range(n_active):
        for _ in range(rng.integers(6, 30)):
            pool = np.flatnonzero(item_type == favourite[u]) if rng.random() < 0.85 else np.arange(n_items)
            day = cutoff_ts - pd.Timedelta(days=int(rng.integers(1, n_weeks * 7)))
            rows.append((u, int(rng.choice(pool)), day))
        for _ in range(rng.integers(0, 3)):                       # test week purchases
            pool = np.flatnonzero(item_type == favourite[u])
            rows.append((u, int(rng.choice(pool)), cutoff_ts + pd.Timedelta(days=int(rng.integers(0, 7)))))
    tx = pd.DataFrame(rows, columns=["u", "i", "t_dat"])
    tx["user_idx"] = (tx["u"] + 1).astype("int32")
    tx["item_idx"] = (tx["i"] + 1).astype("int32")
    tx["price"] = rng.uniform(0.005, 0.08, len(tx))
    tx["sales_channel_id"] = rng.integers(1, 3, len(tx)).astype("int32")
    train = tx[tx["t_dat"] < cutoff_ts].copy()
    test = tx[tx["t_dat"] >= cutoff_ts].copy()
    train["month"] = train["t_dat"].dt.month.astype("int32")
    train["day_of_week"] = ((train["t_dat"].dt.dayofweek + 1) % 7 + 1).astype("int32")
    train["days_ago"] = (cutoff_ts - train["t_dat"]).dt.days.astype("int32")

    cols = ["user_idx", "item_idx", "t_dat", "month", "day_of_week", "days_ago", "price", "sales_channel_id"]
    train[cols].to_parquet(out / "interactions_train.parquet", index=False)
    test[["user_idx", "item_idx", "t_dat", "price", "sales_channel_id"]].to_parquet(
        out / "interactions_test.parquet", index=False)

    # ---------------- user / item tables (same columns as the Spark job)
    g = train.groupby("user_idx")
    feats = pd.DataFrame({"n_purchases": g.size(), "n_unique_items": g["item_idx"].nunique(),
                          "avg_price": g["price"].mean(), "days_since_last_purchase": g["days_ago"].min()})
    users = pd.DataFrame({
        "customer_id": customer_ids, "user_idx": np.arange(1, n_users + 1, dtype="int32"),
        "age": rng.integers(18, 70, n_users).astype(float),
        "club_member_status": rng.choice(["ACTIVE", "PRE-CREATE"], n_users, p=[0.9, 0.1]),
        "fashion_news_frequency": rng.choice(["NONE", "REGULARLY"], n_users), "age_missing": 0,
    }).merge(feats, left_on="user_idx", right_index=True, how="left")
    users["is_new_user"] = users["n_purchases"].isna().astype(int)
    users = users.fillna({"n_purchases": 0, "n_unique_items": 0, "avg_price": 0.0, "days_since_last_purchase": 999})
    users.to_parquet(out / "user_table.parquet", index=False)

    pop = train.groupby("item_idx").agg(
        purchases_last_7d=("days_ago", lambda d: int((d <= 7).sum())),
        purchases_last_28d=("days_ago", lambda d: int((d <= 28).sum())),
        purchases_all_time=("days_ago", "size"))
    items = items.merge(pop, left_on="item_idx", right_index=True, how="left")
    items["is_new_item"] = items["purchases_all_time"].isna().astype(int)
    items = items.fillna({"purchases_last_7d": 0, "purchases_last_28d": 0, "purchases_all_time": 0})
    items.to_parquet(out / "item_table.parquet", index=False)
    users[["customer_id", "user_idx"]].to_parquet(out / "user_vocab.parquet", index=False)
    items[["article_id", "item_idx"]].to_parquet(out / "item_vocab.parquet", index=False)

    vocab = {c: sorted(items[c].unique().tolist()) for c in
             ["product_type_name", "colour_group_name", "department_name", "garment_group_name"]}
    vocab.update({c: sorted(users[c].unique().tolist()) for c in ["club_member_status", "fashion_news_frequency"]})
    (out / "vocabularies.json").write_text(json.dumps(vocab, indent=2))
    top = items.sort_values(["purchases_last_28d", "article_id"], ascending=[False, True])["article_id"]
    (out / "fallback_top12.json").write_text(json.dumps(top.head(12).tolist()))

    # ---------------- extras for the web UI
    details = items[["article_id"]].copy()
    details["prod_name"] = [f"{t} {n}" for t, n in zip(items["product_type_name"], range(n_items))]
    for c in ["product_group_name", "graphical_appearance_name", "index_name", "section_name"]:
        details[c] = "Synthetic"
    details["detail_desc"] = "Synthetic article for tests."
    details.to_parquet(out / "article_details.parquet", index=False)
    last = train.sort_values("t_dat", ascending=False).drop_duplicates(["user_idx", "item_idx"])
    hist = last.groupby("user_idx")["item_idx"].apply(lambda s: ",".join(article_ids[i - 1] for i in s.head(12)))
    pd.DataFrame({"customer_id": [customer_ids[u - 1] for u in hist.index],
                  "recent_articles": hist.values}).to_parquet(out / "user_history.parquet", index=False)
    (out / "build_info.json").write_text(json.dumps({"builder": "synthetic", "images_source": None}))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join("data", "synthetic"))
    args = parser.parse_args()
    print("synthetic data written to", make(args.out))
