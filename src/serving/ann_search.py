"""Week 3, Day 7: approximate nearest neighbour search (FAISS) over item embeddings."""
import json
import time

import faiss
import numpy as np
import redis

EXPORT = "/content/drive/MyDrive/hm_project/export"
K = 10

emb = np.load(f"{EXPORT}/item_embeddings.npy").astype("float32")
ids = json.load(open(f"{EXPORT}/item_ids.json"))
faiss.normalize_L2(emb)          # unit vectors: inner product = cosine similarity
d = emb.shape[1]

# 1) exact search (baseline, always correct)
exact = faiss.IndexFlatIP(d)
exact.add(emb)

# 2) approximate search (HNSW graph index, much faster on big catalogues)
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 200
hnsw.add(emb)
hnsw.hnsw.efSearch = 64

# compare on 2000 random item queries
rng = np.random.default_rng(0)
queries = emb[rng.choice(len(emb), 2000, replace=False)]


def run(index):
    start = time.perf_counter()
    _, found = index.search(queries, K)
    ms = (time.perf_counter() - start) / len(queries) * 1000
    return found, ms


found_exact, ms_exact = run(exact)
found_ann, ms_ann = run(hnsw)
recall = np.mean([len(set(a) & set(b)) / K for a, b in zip(found_exact, found_ann)])

print(f"exact search : {ms_exact:.3f} ms per query")
print(f"HNSW search  : {ms_ann:.3f} ms per query  ({ms_exact / ms_ann:.1f}x faster)")
print(f"ANN recall@{K} vs exact: {recall:.3f}")

faiss.write_index(hnsw, f"{EXPORT}/item_hnsw.index")
print("index saved to", EXPORT)

# demo: take an item vector from Redis, find similar items, print their names
try:
    r = redis.Redis(host="localhost", port=6379)
    item_id = ids[1000]
    vec = np.frombuffer(r.hget(f"item:{item_id}", "vec"), dtype="float32").reshape(1, -1).copy()
    faiss.normalize_L2(vec)
    _, found = hnsw.search(vec, 6)
    print("\nquery item:", item_id, r.hget(f"item:{item_id}", "product_type_name").decode(),
          "/", r.hget(f"item:{item_id}", "colour_group_name").decode())
    for j in found[0][1:]:
        key = f"item:{ids[j]}"
        print("  similar:", ids[j], r.hget(key, "product_type_name").decode(),
              "/", r.hget(key, "colour_group_name").decode())
except Exception as e:
    print("Redis demo skipped:", e)