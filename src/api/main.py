import os
import json
import numpy as np
import pandas as pd
import faiss
import redis

from fastapi import FastAPI, HTTPException, Query

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

EXPORT_PATH = "/content/drive/MyDrive/hm_project/export"

TRANSACTIONS_PATH = (
    "/content/drive/MyDrive/hm_project/transactions_train.csv"
)

# --------------------------------------------------
# 2. LOAD ITEM EMBEDDINGS AND ARTICLE IDs
# --------------------------------------------------

item_embeddings = np.load(
    os.path.join(EXPORT_PATH, "item_embeddings.npy")
).astype("float32")

with open(os.path.join(EXPORT_PATH, "item_ids.json"), "r") as f:
    item_ids = json.load(f)

item_index = {
    article_id: index
    for index, article_id in enumerate(item_ids)
}

# --------------------------------------------------
# 3. LOAD FAISS HNSW INDEX
# --------------------------------------------------

hnsw = faiss.read_index(
    os.path.join(EXPORT_PATH, "item_hnsw.index")
)

# --------------------------------------------------
# 4. LOAD CUSTOMER TRANSACTION HISTORY
# --------------------------------------------------

transactions = pd.read_csv(
    TRANSACTIONS_PATH,
    dtype={
        "customer_id": str,
        "article_id": str
    }
)

transactions["article_id"] = (
    transactions["article_id"].str.zfill(10)
)

# --------------------------------------------------
# 5. CONNECT TO REDIS
# --------------------------------------------------

r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=False
)

# --------------------------------------------------
# 6. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="HM Recommendation API",
    description="Personalized Top-K Product Recommendation System",
    version="1.0"
)

# --------------------------------------------------
# 7. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "HM Recommendation API is running!",
        "status": "success"
    }

# --------------------------------------------------
# 8. TOP-K RECOMMENDATION ENDPOINT
# --------------------------------------------------

@app.get("/recommend/{customer_id}")
def recommend(
    customer_id: str,
    k: int = Query(default=5, ge=1, le=100)
):

    # Find customer transaction history

    history = transactions[
        transactions["customer_id"] == customer_id
    ]

    if history.empty:

        raise HTTPException(
            status_code=404,
            detail="Customer not found"
        )

    # Get unique purchased articles

    purchased_ids = set(history["article_id"])

    # Match purchased articles with available embeddings

    matched_ids = purchased_ids.intersection(item_index)

    if not matched_ids:

        raise HTTPException(
            status_code=422,
            detail="No purchased articles have embeddings"
        )

    # Generate customer preference vector

    purchase_vectors = np.array([
        item_embeddings[item_index[article_id]]
        for article_id in matched_ids
    ])

    customer_vector = purchase_vectors.mean(axis=0)

    customer_vector = customer_vector.astype("float32")

    norm = np.linalg.norm(customer_vector)

    if norm == 0:

        raise HTTPException(
            status_code=422,
            detail="Invalid customer preference vector"
        )

    customer_vector /= norm

    # Search similar products using FAISS HNSW

    query_vector = customer_vector.reshape(1, -1)

    search_k = min(k + len(purchased_ids), hnsw.ntotal)

    recommendations = []

    while True:

        distances, indices = hnsw.search(
            query_vector,
            search_k
        )

        recommendations = [
            item_ids[index]
            for index in indices[0]
            if index != -1
            and item_ids[index] not in purchased_ids
        ][:k]

        if len(recommendations) >= k:
            break

        if search_k >= hnsw.ntotal:
            break

        search_k = min(search_k * 2, hnsw.ntotal)

    # Retrieve product details from Redis

    results = []

    for article_id in recommendations:

        item_data = r.hgetall(
            f"item:{article_id}"
        )

        details = {}

        for key, value in item_data.items():

            field = key.decode(
                "utf-8",
                errors="ignore"
            )

            if field == "vec":
                continue

            details[field] = value.decode(
                "utf-8",
                errors="ignore"
            )

        results.append({
            "article_id": article_id,
            "details": details
        })

    # Return JSON response

    return {
        "customer_id": customer_id,
        "requested_k": k,
        "total_recommendations": len(results),
        "recommendations": results
    }
