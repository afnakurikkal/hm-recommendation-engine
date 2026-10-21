"""Week 3, Days 4-6: Redis as a low-latency feature store.

Loads into Redis:
  user:{customer_id}  hash  model-ready user features (+ readable profile fields
                            and the customer's most recent purchases)
  item:{article_id}   hash  pre-computed item vector (float32 bytes) + metadata
  fallback:top12      str   popular articles for unknown customers (cold start)
  model:info          hash  which model / cutoff the store was built from

Run:  python -m src.serving.load_redis
Env:  REDIS_URL (default redis://localhost:6379/0), HM_MAX_USERS (0 = all customers)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import redis

from src import config
from src.data.features import (apply_standardizer, encode, load_vocabularies,
                               user_numeric_frame)
from src.models.loader import load_model_config, load_trained_users
from src.serving.store import connect

BATCH = 20_000


def _optional_parquet(name, **kwargs):
    path = os.path.join(config.DATA_PATH, name)
    return pd.read_parquet(path, **kwargs) if os.path.exists(path) else None


def select_users(users, max_users):
    """All customers, or a sample: mostly customers with purchases plus some new ones."""
    if not max_users or max_users >= len(users):
        return users
    existing = users[users["is_new_user"] == 0]
    new = users[users["is_new_user"] == 1]
    n_new = min(len(new), max(1, max_users // 20))                # ~5% new customers
    n_existing = min(len(existing), max_users - n_new)
    return pd.concat([existing.sample(n_existing, random_state=0),
                      new.sample(n_new, random_state=0)])


def rebuy_lists(users, days):
    """user_idx -> "a1,a2,..." articles bought in the last `days` days, newest first ("buy again" candidates)."""
    if not days:
        return {}
    inter = pd.read_parquet(os.path.join(config.DATA_PATH, "interactions_train.parquet"),
                            columns=["user_idx", "item_idx", "days_ago"], filters=[("days_ago", "<=", days)])
    inter = inter[inter["user_idx"].isin(set(users["user_idx"])) & (inter["item_idx"] > 0)]
    inter = inter.sort_values("days_ago").drop_duplicates(["user_idx", "item_idx"])
    items = pd.read_parquet(os.path.join(config.DATA_PATH, "item_table.parquet"), columns=["article_id", "item_idx"])
    article = dict(zip(items["item_idx"], items["article_id"]))
    inter["article_id"] = inter["item_idx"].map(article)
    return inter.groupby("user_idx")["article_id"].apply(lambda s: ",".join(s.head(12))).to_dict()


def load_users(r, cfg, vocabularies, max_users=None, log=print):
    users = select_users(pd.read_parquet(os.path.join(config.DATA_PATH, "user_table.parquet")), max_users)

    trained = load_trained_users()
    user_idx = users["user_idx"].to_numpy()
    model_idx = np.where(np.isin(user_idx, list(trained)), user_idx, 0).astype("int32")
    club = encode(users["club_member_status"], vocabularies["club_member_status"])
    news = encode(users["fashion_news_frequency"], vocabularies["fashion_news_frequency"])
    user_num = apply_standardizer(user_numeric_frame(users), cfg["user_num_mean"], cfg["user_num_std"])

    history = _optional_parquet("user_history.parquet")
    recent = {}
    if history is not None:
        history = history[history["customer_id"].isin(set(users["customer_id"]))]
        recent = dict(zip(history["customer_id"], history["recent_articles"]))
    rebuy = rebuy_lists(users, cfg.get("repurchase_days") or 0)

    customer_ids = users["customer_id"].to_numpy()
    ages = users["age"].to_numpy()
    clubs = users["club_member_status"].to_numpy()
    newss = users["fashion_news_frequency"].to_numpy()
    n_purchases = users["n_purchases"].to_numpy()
    avg_price = users["avg_price"].to_numpy()
    last_days = users["days_since_last_purchase"].to_numpy()
    is_new = users["is_new_user"].to_numpy()

    pipe = r.pipeline(transaction=False)
    for row in range(len(users)):
        mapping = {
            "user_idx": int(model_idx[row]),
            "club": int(club[row]),
            "news": int(news[row]),
            "user_num": user_num[row].tobytes(),
            "age": int(ages[row]),
            "club_member_status": str(clubs[row]),
            "fashion_news_frequency": str(newss[row]),
            "n_purchases": int(n_purchases[row]),
            "avg_price": float(avg_price[row]),
            "days_since_last_purchase": int(last_days[row]),
            "is_new_user": int(is_new[row]),
        }
        if customer_ids[row] in recent:
            mapping["recent"] = recent[customer_ids[row]]
        if user_idx[row] in rebuy:
            mapping["rebuy"] = rebuy[user_idx[row]]
        pipe.hset(f"user:{customer_ids[row]}", mapping=mapping)
        if (row + 1) % BATCH == 0:
            pipe.execute()
            print(f"  users loaded: {row + 1:,}/{len(users):,}", end="\r")
    pipe.execute()
    log(f"users loaded: {len(users):,} ({int((is_new == 0).sum()):,} with purchase history)")
    return customer_ids


def load_items(r, log=print):
    embeddings = np.load(config.ITEM_EMBEDDINGS_FILE).astype("float32")
    with open(config.ITEM_IDS_FILE) as f:
        item_ids = json.load(f)
    items = pd.read_parquet(os.path.join(config.DATA_PATH, "item_table.parquet"),
                            columns=["article_id", "purchases_last_28d"] + config.ITEM_CAT_COLS)
    details = _optional_parquet("article_details.parquet")
    if details is not None:
        cols = [c for c in config.ITEM_DETAIL_COLS if c in details.columns]
        items = items.merge(details[["article_id"] + cols], on="article_id", how="left")
    fields = [c for c in items.columns if c != "article_id"]
    meta = items.set_index("article_id")[fields].to_dict("index")

    pipe = r.pipeline(transaction=False)
    for row, article_id in enumerate(item_ids):
        mapping = {"vec": embeddings[row].tobytes()}
        mapping.update({c: str(v) for c, v in meta.get(article_id, {}).items() if v is not None and v == v})
        pipe.hset(f"item:{article_id}", mapping=mapping)
        if (row + 1) % BATCH == 0:
            pipe.execute()
    pipe.execute()
    log(f"items loaded: {len(item_ids):,} (vector size {embeddings.shape[1]})")
    return item_ids


def load_fallback(r, log=print):
    with open(os.path.join(config.DATA_PATH, "fallback_top12.json")) as f:
        top12 = json.load(f)
    r.set("fallback:top12", ",".join(top12))
    log("fallback top-12 stored")


def smoke_test(r, customer_id, article_id, log=print):
    start = time.perf_counter()
    for _ in range(1000):
        r.hgetall(f"user:{customer_id}")
        r.hget(f"item:{article_id}", "vec")
    ms = (time.perf_counter() - start) / 1000 * 1000
    profile = {k.decode(): v for k, v in r.hgetall(f"user:{customer_id}").items() if k != b"user_num"}
    log(f"sample user profile: {profile}")
    log(f"avg lookup (user profile + item vector): {ms:.3f} ms")


def populate(r, max_users=None, log=print):
    """Load the whole feature store. Used by this script and by the API in memory mode."""
    started = time.time()
    cfg = load_model_config()
    vocabularies = load_vocabularies()
    item_ids = load_items(r, log)
    load_fallback(r, log)
    customer_ids = load_users(r, cfg, vocabularies, max_users, log)
    r.hset("model:info", mapping={
        "cutoff": cfg.get("cutoff", config.CUTOFF),
        "embedding_dim": cfg["embedding_dim"],
        "users": len(customer_ids),
        "items": len(item_ids),
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    log(f"feature store loaded in {time.time() - started:.0f}s")
    return customer_ids, item_ids


def main():
    started = time.time()
    r = connect()
    max_users = int(os.environ.get("HM_MAX_USERS", "0")) or None
    customer_ids, item_ids = populate(r, max_users)
    smoke_test(r, customer_ids[0], item_ids[0])
    try:
        memory = r.info("memory")["used_memory_human"]
    except (redis.RedisError, KeyError):
        memory = "n/a"
    print("keys in Redis:", r.dbsize(), "| memory:", memory)
    print(f"done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
