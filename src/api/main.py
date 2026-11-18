"""Week 4, Days 1-3: FastAPI recommendation microservice.

Flow for GET /recommend/{customer_id}:
  1. fetch the customer's features from Redis (feature store)
  2. query tower -> customer embedding (with month / day-of-week context)
  3. FAISS HNSW (ANN) search over the pre-computed item embeddings
  4. fetch article details from Redis and return the top-K
Unknown customers get the popular-items fallback (cold start).

Run:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
import json
import time
from contextlib import asynccontextmanager

import faiss
import numpy as np
import redis
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query

from src import config
from src.data.features import context_features
from src.models.loader import load_model_config, load_query_tower

STATE = {}


def load_index():
    STATE["index"] = faiss.read_index(config.ANN_INDEX_FILE)
    with open(config.ITEM_IDS_FILE) as f:
        STATE["item_ids"] = json.load(f)


def load_everything():
    cfg = load_model_config()
    tower = load_query_tower(cfg)

    @tf.function(reduce_retracing=True)
    def embed(inputs):
        return tower(inputs, training=False)

    STATE.update(cfg=cfg, embed=embed, context=context_features(),
                 redis=redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT))
    load_index()
    STATE["redis"].ping()
    embed_user({b"user_idx": b"0", b"club": b"0", b"news": b"0",
                b"user_num": np.zeros(len(cfg["user_num_cols"]), "float32").tobytes()})  # warm-up


@asynccontextmanager
async def lifespan(app):
    load_everything()
    yield
    STATE.clear()


app = FastAPI(
    title="H&M Recommendation API",
    description="Context-aware two-tower recommendations served from Redis + FAISS",
    version="1.0",
    lifespan=lifespan,
)


def embed_user(profile):
    month, day_of_week = STATE["context"]
    inputs = {
        "user_idx": tf.constant([int(profile[b"user_idx"])], tf.int32),
        "club": tf.constant([int(profile[b"club"])], tf.int32),
        "news": tf.constant([int(profile[b"news"])], tf.int32),
        "month": tf.constant([month], tf.int32),
        "day_of_week": tf.constant([day_of_week], tf.int32),
        "user_num": tf.constant(np.frombuffer(profile[b"user_num"], dtype="float32")[None, :]),
    }
    return STATE["embed"](inputs).numpy().astype("float32")


def item_details(article_ids):
    pipe = STATE["redis"].pipeline(transaction=False)
    for article_id in article_ids:
        pipe.hmget(f"item:{article_id}", config.ITEM_CAT_COLS)
    rows = pipe.execute()
    return [
        {"article_id": a, **{c: (v.decode() if v else None) for c, v in zip(config.ITEM_CAT_COLS, row)}}
        for a, row in zip(article_ids, rows)
    ]


def popular_items(k):
    raw = STATE["redis"].get("fallback:top12")
    return raw.decode().split(",")[:k] if raw else []


@app.get("/")
def home():
    return {"message": "H&M Recommendation API is running!", "status": "success", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        redis_ok = bool(STATE["redis"].ping())
    except redis.RedisError:
        redis_ok = False
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
        "items_in_index": int(STATE["index"].ntotal),
        "embedding_dim": STATE["cfg"]["embedding_dim"],
        "context_month_day_of_week": STATE["context"],
    }


@app.get("/recommend/{customer_id}")
def recommend(customer_id: str, k: int = Query(default=12, ge=1, le=100)):
    started = time.perf_counter()
    profile = STATE["redis"].hgetall(f"user:{customer_id}")

    if profile:
        _, rows = STATE["index"].search(embed_user(profile), k)
        article_ids = [STATE["item_ids"][row] for row in rows[0] if row >= 0]
        source = "two_tower_model"
        customer = {
            "age": int(profile.get(b"age", b"0")),
            "club_member_status": profile.get(b"club_member_status", b"").decode(),
            "n_purchases": int(profile.get(b"n_purchases", b"0")),
            "is_new_customer": profile.get(b"is_new_user", b"0") == b"1",
        }
    else:
        article_ids = popular_items(k)
        source = "popular_fallback"
        customer = None

    return {
        "customer_id": customer_id,
        "source": source,
        "customer": customer,
        "k": k,
        "recommendations": item_details(article_ids),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.get("/similar/{article_id}")
def similar(article_id: str, k: int = Query(default=12, ge=1, le=100)):
    """Items similar to an article (item-to-item ANN search on the item vectors)."""
    started = time.perf_counter()
    raw = STATE["redis"].hget(f"item:{article_id}", "vec")
    if raw is None:
        raise HTTPException(status_code=404, detail="Article not found")
    vector = np.frombuffer(raw, dtype="float32")[None, :].copy()
    faiss.normalize_L2(vector)
    _, rows = STATE["index"].search(vector, k + 1)
    article_ids = [STATE["item_ids"][r] for r in rows[0] if r >= 0 and STATE["item_ids"][r] != article_id][:k]
    return {
        "article_id": article_id,
        "similar": item_details(article_ids),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.post("/admin/reload")
def reload_index():
    """Reload the ANN index after the daily embedding refresh (called by Airflow)."""
    load_index()
    return {"status": "reloaded", "items_in_index": int(STATE["index"].ntotal)}
