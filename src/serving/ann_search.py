"""Week 3, Day 7: Approximate Nearest Neighbour (ANN) search over item embeddings.

Builds a FAISS HNSW index from the item embeddings exported in Week 2, checks
its accuracy and speed against exact search, and saves it for the API.

Run:  python -m src.serving.ann_search
"""
import json
import time

import faiss
import numpy as np

from src import config

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
    print("item embeddings:", embeddings.shape)

    started = time.time()
    hnsw = build_hnsw(embeddings)
    print(f"HNSW index built in {time.time() - started:.1f}s")

    exact = faiss.IndexFlatIP(embeddings.shape[1])
    exact.add(embeddings)
    rng = np.random.default_rng(0)
    queries = embeddings[rng.choice(len(embeddings), min(2000, len(embeddings)), replace=False)]
    ms_exact, ms_ann, recall = benchmark(exact, hnsw, queries)
    print(f"exact search : {ms_exact:.3f} ms per query")
    print(f"HNSW search  : {ms_ann:.3f} ms per query ({ms_exact / max(ms_ann, 1e-9):.1f}x faster)")
    print(f"ANN recall@{K} vs exact search: {recall:.3f}")

    faiss.write_index(hnsw, config.ANN_INDEX_FILE)
    print("index saved to", config.ANN_INDEX_FILE)


if __name__ == "__main__":
    main()
