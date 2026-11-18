"""Week 3, Days 4-6: Redis as a low-latency feature store.

Loads into Redis:
  user:{customer_id}  hash  model-ready user features (+ readable profile fields)
  item:{article_id}   hash  pre-computed item vector (float32 bytes) + metadata
  fallback:top12      str   popular articles for unknown customers (cold start)
  model:info          hash  which model / cutoff the store was built from

Run:  python -m src.serving.load_redis
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

BATCH = 20_000


def connect():
    r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)
    r.ping()
    return r


def load_users(r, cfg, vocabularies, max_users=None):
    users = pd.read_parquet(os.path.join(config.DATA_PATH, "user_table.parquet"))
    if max_users:
        # keep customers with purchases first so the demo has useful profiles
        users = users.sort_values("is_new_user").head(max_users)

    trained = load_trained_users()
    user_idx = users["user_idx"].to_numpy()
    model_idx = np.where(np.isin(user_idx, list(trained)), user_idx, 0).astype("int32")
    club = encode(users["club_member_status"], vocabularies["club_member_status"])
    news = encode(users["fashion_news_frequency"], vocabularies["fashion_news_frequency"])
    user_num = apply_standardizer(user_numeric_frame(users), cfg["user_num_mean"], cfg["user_num_std"])

    customer_ids = users["customer_id"].to_numpy()
    ages = users["age"].to_numpy()
    clubs = users["club_member_status"].to_numpy()
    newss = users["fashion_news_frequency"].to_numpy()
    n_purchases = users["n_purchases"].to_numpy()
    is_new = users["is_new_user"].to_numpy()

    pipe = r.pipeline(transaction=False)
    for row in range(len(users)):
        pipe.hset(f"user:{customer_ids[row]}", mapping={
            "user_idx": int(model_idx[row]),
            "club": int(club[row]),
            "news": int(news[row]),
            "user_num": user_num[row].tobytes(),
            "age": int(ages[row]),
            "club_member_status": str(clubs[row]),
            "fashion_news_frequency": str(newss[row]),
            "n_purchases": int(n_purchases[row]),
            "is_new_user": int(is_new[row]),
        })
        if (row + 1) % BATCH == 0:
            pipe.execute()
            print(f"  users loaded: {row + 1:,}/{len(users):,}", end="\r")
    pipe.execute()
    print(f"users loaded: {len(users):,}                ")
    return customer_ids


def load_items(r):
    embeddings = np.load(config.ITEM_EMBEDDINGS_FILE).astype("float32")
    with open(config.ITEM_IDS_FILE) as f:
        item_ids = json.load(f)
    items = pd.read_parquet(os.path.join(config.DATA_PATH, "item_table.parquet"),
                            columns=["article_id"] + config.ITEM_CAT_COLS)
    meta = items.set_index("article_id")[config.ITEM_CAT_COLS].to_dict("index")

    pipe = r.pipeline(transaction=False)
    for row, article_id in enumerate(item_ids):
        mapping = {"vec": embeddings[row].tobytes()}
        mapping.update({c: str(v) for c, v in meta.get(article_id, {}).items()})
        pipe.hset(f"item:{article_id}", mapping=mapping)
        if (row + 1) % BATCH == 0:
            pipe.execute()
    pipe.execute()
    print(f"items loaded: {len(item_ids):,} (vector size {embeddings.shape[1]})")
    return item_ids


def load_fallback(r):
    with open(os.path.join(config.DATA_PATH, "fallback_top12.json")) as f:
        top12 = json.load(f)
    r.set("fallback:top12", ",".join(top12))
    print("fallback top-12 stored")


def smoke_test(r, customer_id, article_id):
    start = time.perf_counter()
    for _ in range(1000):
        r.hgetall(f"user:{customer_id}")
        r.hget(f"item:{article_id}", "vec")
    ms = (time.perf_counter() - start) / 1000 * 1000
    profile = {k.decode(): v for k, v in r.hgetall(f"user:{customer_id}").items() if k != b"user_num"}
    print("sample user profile:", profile)
    print(f"avg lookup (user profile + item vector): {ms:.3f} ms")


def main():
    started = time.time()
    r = connect()
    cfg = load_model_config()
    vocabularies = load_vocabularies()
    max_users = int(os.environ.get("HM_MAX_USERS", "0")) or None

    item_ids = load_items(r)
    load_fallback(r)
    customer_ids = load_users(r, cfg, vocabularies, max_users)
    r.hset("model:info", mapping={
        "cutoff": cfg.get("cutoff", config.CUTOFF),
        "embedding_dim": cfg["embedding_dim"],
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    smoke_test(r, customer_ids[0], item_ids[0])
    try:
        memory = r.info("memory")["used_memory_human"]
    except redis.RedisError:
        memory = "n/a"
    print("keys in Redis:", r.dbsize(), "| memory:", memory)
    print(f"done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
