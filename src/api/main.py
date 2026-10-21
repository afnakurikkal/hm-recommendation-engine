"""Week 4, Days 1-3: FastAPI recommendation microservice.

Flow for GET /recommend/{customer_id}:
  1. fetch the customer's features from Redis (feature store)
  2. query tower -> customer embedding (with month / day-of-week context)
  3. FAISS HNSW (ANN) search over the pre-computed item embeddings
  4. fetch article details from Redis and return the top-K
Unknown customers get the popular-items fallback (cold start).

Run:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Web UI: http://localhost:8000/ui      API docs: http://localhost:8000/docs
"""
import io
import json
import logging
import os
import threading
import time
import zipfile
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date as Date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import redis
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from src import config
from src.data.features import context_features
from src.models.loader import load_model_config, load_query_tower
from src.serving import store

log = logging.getLogger("hm_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATE = {}
UI_FILE = Path(__file__).parent / "static" / "index.html"
DETAIL_FIELDS = config.ITEM_CAT_COLS + config.ITEM_DETAIL_COLS + ["purchases_last_28d"]
LATENCIES = defaultdict(lambda: deque(maxlen=2000))     # endpoint -> recent latencies (ms)
COUNTS = defaultdict(int)
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


# --------------------------------------------------------------------------- loading
def load_index():
    STATE["index"] = faiss.read_index(config.ANN_INDEX_FILE)
    with open(config.ITEM_IDS_FILE) as f:
        STATE["item_ids"] = json.load(f)
    vectors = np.load(config.ITEM_EMBEDDINGS_FILE).astype("float32")
    faiss.normalize_L2(vectors)
    STATE["item_vecs"] = vectors                         # for the cosine "match" score shown to users
    STATE["row_of"] = {article_id: row for row, article_id in enumerate(STATE["item_ids"])}
    # one extra index dimension = recent-popularity prior (see src/serving/ann_search.py)
    STATE["ranked"] = STATE["index"].d == vectors.shape[1] + 1


def find_images_source():
    """Kaggle zip / folder with product photos (env HM_IMAGES or the data build info)."""
    source = config.IMAGES_SOURCE
    info_file = Path(config.DATA_PATH) / "build_info.json"
    if not source and info_file.exists():
        source = json.loads(info_file.read_text()).get("images_source")
    if not source or not Path(source).exists():
        return None
    if Path(source).is_file():
        return {"zip": zipfile.ZipFile(source), "lock": threading.Lock()}
    folder = Path(source) / "images" if (Path(source) / "images").is_dir() else Path(source)
    return {"dir": folder}


def load_everything():
    started = time.time()
    cfg = load_model_config()
    tower = load_query_tower(cfg)

    @tf.function(reduce_retracing=True)
    def embed(inputs):
        return tower(inputs, training=False)

    STATE.update(cfg=cfg, embed=embed, context=context_features(), redis=store.connect())
    load_index()
    if store.is_memory():
        # no Redis server: build the feature store inside this process
        from src.serving.load_redis import populate
        populate(STATE["redis"], int(os.environ.get("HM_MAX_USERS", "50000")) or None, log=log.info)
    STATE["images"] = find_images_source()
    embed_user({b"user_idx": b"0", b"club": b"0", b"news": b"0",
                b"user_num": np.zeros(len(cfg["user_num_cols"]), "float32").tobytes()})   # warm-up
    STATE["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log.info("model, ANN index (%d items) and feature store ready in %.1fs",
             STATE["index"].ntotal, time.time() - started)


@asynccontextmanager
async def lifespan(app):
    try:
        load_everything()
    except Exception as exc:  # noqa: BLE001 - missing files, Redis down, faiss errors ...
        # Keep the server up so /health and the web UI can explain what is missing.
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log.error("API started WITHOUT a model: %s", STATE["error"])
        log.error("Build the data, train the model and load the feature store first (see README: Quick start).")
    yield
    STATE.clear()


app = FastAPI(
    title="H&M Recommendation API",
    description=(
        "Context-aware two-tower neural recommendations for the H&M Personalized Fashion dataset.\n\n"
        "Customer features come from the **Redis feature store**, the **query tower** (TensorFlow) turns them "
        "plus the purchase context (month, day of week) into a vector, and **FAISS HNSW** finds the nearest "
        "article vectors. Web UI: [/ui](/ui)"
    ),
    version="2.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - started) * 1000
    route = request.scope.get("route")
    if route is not None and request.url.path not in ("/stats", "/health") and not request.url.path.startswith("/images"):
        LATENCIES[route.path].append(ms)
        COUNTS[route.path] += 1
    response.headers["X-Response-Time-ms"] = f"{ms:.2f}"
    return response


def ready():
    if "index" not in STATE:
        raise HTTPException(status_code=503, detail=f"Model not loaded - {STATE.get('error', 'starting up')}")


# --------------------------------------------------------------------------- helpers
def parse_context(day: Optional[Date]):
    """(month, day_of_week) for the given date, or the default context date."""
    if day is None:
        return STATE["context"]
    return context_features(day.isoformat())


def embed_user(profile, context=None):
    month, day_of_week = context or STATE["context"]
    inputs = {
        "user_idx": tf.constant([int(profile[b"user_idx"])], tf.int32),
        "club": tf.constant([int(profile[b"club"])], tf.int32),
        "news": tf.constant([int(profile[b"news"])], tf.int32),
        "month": tf.constant([month], tf.int32),
        "day_of_week": tf.constant([day_of_week], tf.int32),
        "user_num": tf.constant(np.frombuffer(profile[b"user_num"], dtype="float32")[None, :]),
    }
    return STATE["embed"](inputs).numpy().astype("float32")


def item_details(article_ids, scores=None, extra=None):
    pipe = STATE["redis"].pipeline(transaction=False)
    for article_id in article_ids:
        pipe.hmget(f"item:{article_id}", DETAIL_FIELDS)
    rows = pipe.execute()
    images = STATE.get("images") is not None
    out = []
    for n, (article_id, row) in enumerate(zip(article_ids, rows)):
        item = {"article_id": article_id, **{c: (v.decode() if v else None) for c, v in zip(DETAIL_FIELDS, row)}}
        item["purchases_last_28d"] = int(float(item["purchases_last_28d"] or 0))
        item["image_url"] = f"/images/{article_id}" if images else None
        if scores is not None:
            item["score"] = round(float(scores[n]), 4)
        if extra is not None:
            item.update(extra[n])
        out.append(item)
    return out


def popular_items(k):
    raw = STATE["redis"].get("fallback:top12")
    return raw.decode().split(",")[:k] if raw else []


def ann_search(vector, k, exclude=(), personalised=True):
    """Top-k article ids (and cosine scores) for a query vector, skipping `exclude`.

    personalised=True ranks by cosine + popularity prior (when the index has it);
    False (item-to-item) ranks by cosine only.
    """
    if k <= 0:
        return [], [], []
    exclude = set(exclude)
    query = vector
    if STATE["ranked"]:
        query = np.hstack([vector, np.full((1, 1), 1.0 if personalised else 0.0, "float32")])
    distances, rows = STATE["index"].search(query, k + len(exclude))
    found = [(r, d) for r, d in zip(rows[0], distances[0]) if r >= 0 and STATE["item_ids"][r] not in exclude][:k]
    rows = [r for r, _ in found]
    cosine = STATE["item_vecs"][rows] @ vector[0]
    return [STATE["item_ids"][r] for r in rows], [float(s) for s in cosine], [float(d) for _, d in found]


def cosine_scores(vector, article_ids):
    rows = [STATE["row_of"].get(a) for a in article_ids]
    return [float(STATE["item_vecs"][r] @ vector[0]) if r is not None else 0.0 for r in rows]


def rebuy_candidates(profile):
    """Articles the customer bought in the last `repurchase_days` days (newest first; see tune_ranking.py)."""
    raw = profile.get(b"rebuy")
    return raw.decode().split(",") if raw and STATE["cfg"].get("repurchase_days") else []


def customer_profile(profile):
    def get(field, default=b"0"):
        return profile.get(field.encode(), default)
    return {
        "age": int(get("age")),
        "club_member_status": get("club_member_status", b"").decode(),
        "fashion_news_frequency": get("fashion_news_frequency", b"").decode(),
        "n_purchases": int(get("n_purchases")),
        "avg_price": round(float(get("avg_price")), 4),
        "days_since_last_purchase": int(get("days_since_last_purchase", b"999")),
        "is_new_customer": get("is_new_user") == b"1",
        "has_trained_embedding": int(get("user_idx")) > 0,
    }


def recent_purchases(profile):
    raw = profile.get(b"recent")
    return raw.decode().split(",") if raw else []


# --------------------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def home(request: Request):
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse("/ui")
    return {"message": "H&M Recommendation API is running!", "status": "success",
            "frontend": "/ui", "docs": "/docs"}


@app.get("/health", tags=["service"])
def health():
    """Service status: model, ANN index and feature store."""
    if "index" not in STATE:
        return {"status": "no_model", "redis": False, "items_in_index": 0, "error": STATE.get("error")}
    try:
        redis_ok = bool(STATE["redis"].ping())
        users = int((STATE["redis"].hget("model:info", "users") or b"0"))
    except redis.RedisError:
        redis_ok, users = False, 0
    month, dow = STATE["context"]
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
        "feature_store": "in-memory" if store.is_memory() else "redis",
        "items_in_index": int(STATE["index"].ntotal),
        "customers_in_store": users,
        "embedding_dim": STATE["cfg"]["embedding_dim"],
        "context_month_day_of_week": STATE["context"],
        "context": {"date": config.CONTEXT_DATE, "month": month, "day_of_week": DAYS[dow - 1]},
        "images": STATE.get("images") is not None,
        "loaded_at": STATE.get("loaded_at"),
    }


@app.get("/model", tags=["service"])
def model_info():
    """Model settings, offline test metrics (Recall@12, NDCG@12, MAP@12) and training history."""
    ready()
    cfg = STATE["cfg"]
    keys = ["embedding_dim", "id_dim", "temperature", "id_dropout", "learning_rate", "batch_size",
            "epochs_run", "best_epoch", "best_val_loss", "cutoff", "train_weeks", "n_users", "n_items",
            "train_interactions", "test_customers", "trained_at", "training_minutes", "popularity_weight",
            "repurchase_days"]
    bench_file = Path(config.MODEL_DIR) / "ann_benchmark.json"
    return {**{k: cfg.get(k) for k in keys},
            "ranking": "two-tower + recent popularity prior" if STATE["ranked"] else "two-tower",
            "ann_benchmark": json.loads(bench_file.read_text()) if bench_file.exists() else None,
            "test_metrics": cfg.get("test_metrics"), "history": cfg.get("history")}


@app.get("/recommend/{customer_id}", tags=["recommendations"])
def recommend(
    customer_id: str,
    k: int = Query(default=12, ge=1, le=100, description="number of articles"),
    date: Optional[Date] = Query(default=None, description="context date (YYYY-MM-DD): month and day of week "
                                                           "are fed to the query tower"),
    exclude_purchased: bool = Query(default=False, description="hide articles the customer bought recently"),
):
    """Top-K personalised articles for a customer (popular items for unknown customers)."""
    ready()
    started = time.perf_counter()
    context = parse_context(date)
    profile = STATE["redis"].hgetall(f"user:{customer_id}")

    extra = None
    if profile:
        vector = embed_user(profile, context)
        bought = recent_purchases(profile)
        # 1. "buy again": articles bought in the last few weeks; 2. two-tower model fills the rest
        rebuy = [] if exclude_purchased else rebuy_candidates(profile)[:k]
        exclude = set(bought if exclude_purchased else []) | set(rebuy)
        model_ids, model_scores, rank_scores = ann_search(vector, k - len(rebuy), exclude)
        article_ids = rebuy + model_ids
        scores = cosine_scores(vector, rebuy) + model_scores
        extra = ([{"reason": "bought_recently", "rank_score": None} for _ in rebuy] +
                 [{"reason": "two_tower_model", "rank_score": round(s, 4)} for s in rank_scores])
        source, customer = "two_tower_model", customer_profile(profile)
    else:
        article_ids, scores = popular_items(k), None
        source, customer = "popular_fallback", None

    return {
        "customer_id": customer_id,
        "source": source,
        "customer": customer,
        "context": {"month": context[0], "day_of_week": DAYS[context[1] - 1],
                    "date": date.isoformat() if date else config.CONTEXT_DATE},
        "k": k,
        "recommendations": item_details(article_ids, scores, extra),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.get("/customers/{customer_id}", tags=["recommendations"])
def customer(customer_id: str):
    """Customer profile from the feature store and their most recent purchases."""
    ready()
    profile = STATE["redis"].hgetall(f"user:{customer_id}")
    if not profile:
        raise HTTPException(status_code=404, detail="Customer not in the feature store")
    return {"customer_id": customer_id, "profile": customer_profile(profile),
            "recent_purchases": item_details(recent_purchases(profile))}


@app.get("/similar/{article_id}", tags=["recommendations"])
def similar(article_id: str, k: int = Query(default=12, ge=1, le=100)):
    """Items similar to an article (item-to-item ANN search on the learned item vectors)."""
    ready()
    started = time.perf_counter()
    raw = STATE["redis"].hget(f"item:{article_id}", "vec")
    if raw is None:
        raise HTTPException(status_code=404, detail="Article not found")
    vector = np.frombuffer(raw, dtype="float32")[None, :].copy()
    faiss.normalize_L2(vector)
    article_ids, scores, _ = ann_search(vector, k, exclude=[article_id], personalised=False)
    return {
        "article_id": article_id,
        "article": item_details([article_id])[0],
        "similar": item_details(article_ids, scores),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.get("/articles/{article_id}", tags=["recommendations"])
def article(article_id: str):
    """Article metadata from the feature store."""
    ready()
    if not STATE["redis"].exists(f"item:{article_id}"):
        raise HTTPException(status_code=404, detail="Article not found")
    return item_details([article_id])[0]


@lru_cache(maxsize=4096)
def _thumbnail(article_id):
    images = STATE.get("images")
    name = f"{article_id[:3]}/{article_id}.jpg"
    try:
        if "zip" in images:
            with images["lock"]:
                data = images["zip"].read(f"images/{name}")
        else:
            data = (images["dir"] / name).read_bytes()
    except (KeyError, FileNotFoundError):
        return None
    try:                                     # shrink the 1-2 MB originals for the web UI
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((360, 480))
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=82, optimize=True)
        return buffer.getvalue()
    except ImportError:
        return data


@app.get("/images/{article_id}", include_in_schema=False)
def image(article_id: str):
    if not STATE.get("images") or not article_id.isdigit():
        raise HTTPException(status_code=404, detail="No image")
    data = _thumbnail(article_id)
    if data is None:
        raise HTTPException(status_code=404, detail="No image for this article")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """Web frontend (single page that calls this API)."""
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


@app.get("/sample-customers", tags=["service"])
def sample_customers():
    """Example customers for demos: one with purchase history, one without, and a few more."""
    ready()
    if "samples" in STATE:
        return STATE["samples"]
    found, examples = {}, []
    for key in STATE["redis"].scan_iter("user:*", count=2000):
        r = STATE["redis"]
        is_new, n, recent = r.hmget(key, ["is_new_user", "n_purchases", "recent"])
        customer_id = key.decode().split(":", 1)[1]
        if is_new == b"1":
            found.setdefault("new", customer_id)
        elif int(n or 0) >= 8 and recent and recent.count(b",") >= 5:
            found.setdefault("existing", customer_id)
            if len(examples) < 6:
                examples.append(customer_id)
        if "new" in found and len(examples) >= 6:
            break
    found["examples"] = examples
    STATE["samples"] = found
    return found


@app.get("/stats", tags=["service"])
def stats():
    """Request counts and latency percentiles (ms) per endpoint since start-up."""
    out = {}
    for path, values in LATENCIES.items():
        arr = np.array(values)
        out[path] = {"requests": COUNTS[path], "p50_ms": round(float(np.percentile(arr, 50)), 2),
                     "p95_ms": round(float(np.percentile(arr, 95)), 2),
                     "p99_ms": round(float(np.percentile(arr, 99)), 2)}
    return out


@app.post("/admin/reload", tags=["service"])
def reload_index():
    """Reload the ANN index after the daily embedding refresh (called by Airflow)."""
    ready()
    load_index()
    STATE.pop("samples", None)
    return {"status": "reloaded", "items_in_index": int(STATE["index"].ntotal)}
