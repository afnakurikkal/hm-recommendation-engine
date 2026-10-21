"""Week 3, Day 7: Approximate Nearest Neighbour (ANN) search over item embeddings.

Builds a FAISS HNSW index from the item embeddings exported in Week 2, checks
its accuracy and speed against exact search, and saves it for the API.

Ranking stage in the index: when `popularity_weight` is set in model_config.json
(src/training/tune_ranking.py), every item vector gets one extra dimension

    item  = [embedding, w * T * (log P_recent(item) - max)]      query = [user_vector, 1]

so one inner-product search returns the top items for
cosine(user, item) + w * T * log P_recent(item), the same ranking as
cosine / T + w * log P_recent (T = softmax temperature). Item-to-item search uses
query = [item_vector, 0], which is plain cosine similarity. The vectors stay
close to unit length, so HNSW keeps its high recall.

Run:  python -m src.serving.ann_search
"""
import json
import os
import time

import faiss
import numpy as np
import pandas as pd

from src import config
from src.models.loader import load_model_config

K = 12


def build_hnsw(embeddings, m=32, ef_construction=200, ef_search=128):
    index = faiss.IndexHNSWFlat(embeddings.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    index.hnsw.efSearch = ef_search
    return index


def load_embeddings():
    embeddings = np.load(config.ITEM_EMBEDDINGS_FILE).astype("float32")
    faiss.normalize_L2(embeddings)      # unit vectors: inner product = cosine similarity
    with open(config.ITEM_IDS_FILE) as f:
        item_ids = json.load(f)
    return embeddings, item_ids


def recent_log_prior(item_ids):
    """log share of each item in the purchases of the last 7 days (rows in item_ids order)."""
    items = pd.read_parquet(os.path.join(config.DATA_PATH, "item_table.parquet"),
                            columns=["article_id", "purchases_last_7d"])
    counts = items.set_index("article_id")["purchases_last_7d"].reindex(item_ids).fillna(0).to_numpy("float64")
    return np.log((counts + 1) / (counts.sum() + len(counts))).astype("float32")


def index_vectors(embeddings, item_ids, cfg=None):
    """Item vectors as stored in the index (with the popularity dimension when enabled)."""
    cfg = cfg if cfg is not None else load_model_config()
    weight = cfg.get("popularity_weight") or 0.0
    if not weight:
        return embeddings
    prior = recent_log_prior(item_ids)
    extra = weight * cfg["temperature"] * (prior - prior.max())
    return np.hstack([embeddings, extra[:, None]]).astype("float32")


def benchmark(exact, ann, queries):
    def timed(index):
        start = time.perf_counter()
        _, found = index.search(queries, K)
        return found, (time.perf_counter() - start) / len(queries) * 1000

    found_exact, ms_exact = timed(exact)
    found_ann, ms_ann = timed(ann)
    recall = np.mean([len(set(a) & set(b)) / K for a, b in zip(found_exact, found_ann)])
    return ms_exact, ms_ann, recall


def main():
    embeddings, item_ids = load_embeddings()
    vectors = index_vectors(embeddings, item_ids)
    print("item embeddings:", embeddings.shape,
          "| popularity prior dimension:", "yes" if vectors.shape[1] > embeddings.shape[1] else "no")

    started = time.time()
    hnsw = build_hnsw(vectors)
    print(f"HNSW index built in {time.time() - started:.1f}s")

    exact = faiss.IndexFlatIP(vectors.shape[1])
    exact.add(vectors)
    rng = np.random.default_rng(0)
    queries = embeddings[rng.choice(len(embeddings), min(2000, len(embeddings)), replace=False)]
    if vectors.shape[1] > embeddings.shape[1]:                  # user-style queries: [vector, 1]
        queries = np.hstack([queries, np.ones((len(queries), 1), "float32")])
    ms_exact, ms_ann, recall = benchmark(exact, hnsw, queries)
    print(f"exact search : {ms_exact:.3f} ms per query")
    print(f"HNSW search  : {ms_ann:.3f} ms per query ({ms_exact / max(ms_ann, 1e-9):.1f}x faster)")
    print(f"ANN recall@{K} vs exact search: {recall:.3f}")

    faiss.write_index(hnsw, config.ANN_INDEX_FILE)
    with open(os.path.join(config.MODEL_DIR, "ann_benchmark.json"), "w") as f:
        json.dump({"items": len(item_ids), "dim": int(vectors.shape[1]), "exact_ms_per_query": ms_exact,
                   "hnsw_ms_per_query": ms_ann, "speedup": ms_exact / max(ms_ann, 1e-9),
                   f"recall@{K}_vs_exact": float(recall)}, f, indent=2)
    print("index saved to", config.ANN_INDEX_FILE)


if __name__ == "__main__":
    main()
