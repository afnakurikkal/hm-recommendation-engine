import numpy as np


def generate_recommendations(
    customer_id,
    transactions,
    item_embeddings,
    item_ids,
    item_index,
    hnsw,
    k=5
):
    # Get customer's purchase history
    history = transactions[
        transactions["customer_id"] == customer_id
    ]

    if history.empty:
        raise ValueError("Customer not found")

    # Identify previously purchased products
    purchased_ids = set(history["article_id"])

    # Match purchased products with available embeddings
    matched_ids = purchased_ids.intersection(item_index)

    if not matched_ids:
        raise ValueError("No purchased articles have embeddings")

    # Generate customer preference vector
    purchase_vectors = np.array([
        item_embeddings[item_index[article_id]]
        for article_id in matched_ids
    ])

    customer_vector = purchase_vectors.mean(axis=0)
    customer_vector = customer_vector.astype("float32")

    norm = np.linalg.norm(customer_vector)

    if norm == 0:
        raise ValueError("Invalid customer preference vector")

    customer_vector /= norm
    query_vector = customer_vector.reshape(1, -1)

    # Search similar products using FAISS
    search_k = min(k + len(purchased_ids), hnsw.ntotal)

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

    return recommendations
