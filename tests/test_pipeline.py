"""End-to-end test of the whole system on a small synthetic dataset (no Redis, no Kaggle data).

    python -m pytest tests/test_pipeline.py -v

Steps: Week 1 style data -> Week 2 training + evaluation -> Week 3 ANN index +
feature store -> Week 4 API (FastAPI TestClient). Takes about a minute on a laptop.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def client():
    from scripts.make_synthetic_data import make
    from src import config

    make(config.DATA_PATH)
    from src.training import train_two_tower
    train_two_tower.main()
    from src.training import tune_ranking
    tune_ranking.main()
    from src.serving import ann_search
    ann_search.main()

    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def users():
    from src import config
    return pd.read_parquet(os.path.join(config.DATA_PATH, "user_table.parquet"))


def existing_customer(users):
    return users[users["is_new_user"] == 0].iloc[0]["customer_id"]


def test_training_exports_model_and_report(client):
    from src import config
    for path in [config.QUERY_WEIGHTS_FILE, config.CANDIDATE_WEIGHTS_FILE, config.ITEM_EMBEDDINGS_FILE,
                 config.ANN_INDEX_FILE, config.MODEL_CONFIG_FILE]:
        assert os.path.exists(path), path
    report = Path(config.ARTIFACTS_DIR) / "EVALUATION_REPORT.md"
    assert report.exists() and "Recall@12" in report.read_text(encoding="utf-8")


def test_model_beats_popularity_baseline(client):
    metrics = client.get("/model").json()["test_metrics"]
    assert metrics["two_tower_model"]["Recall@12"] > metrics["popularity_baseline"]["Recall@12"]


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["redis"] is True
    assert data["items_in_index"] > 0
    assert data["feature_store"] == "in-memory"


def test_home_returns_json_for_api_clients(client):
    assert client.get("/").json()["status"] == "success"


def test_recommend_existing_customer(client, users):
    data = client.get(f"/recommend/{existing_customer(users)}", params={"k": 12}).json()
    recs = data["recommendations"]
    assert data["source"] == "two_tower_model"
    assert len(recs) == 12 and len({r["article_id"] for r in recs}) == 12
    assert all(r["product_type_name"] for r in recs)
    ranked = [r["rank_score"] for r in recs if r["reason"] == "two_tower_model"]
    assert ranked and ranked == sorted(ranked, reverse=True)
    assert all(-1.01 <= r["score"] <= 1.01 for r in recs)              # cosine "match" shown in the UI
    assert data["customer"]["n_purchases"] > 0


def test_recommendations_follow_customer_taste(client, users):
    """Synthetic customers buy mostly one product type: the model must pick that up."""
    matches = 0
    sample = users[users["is_new_user"] == 0].head(20)["customer_id"]
    for customer_id in sample:
        history = client.get(f"/customers/{customer_id}").json()["recent_purchases"]
        favourite = Counter(i["product_type_name"] for i in history).most_common(1)[0][0]
        recs = client.get(f"/recommend/{customer_id}", params={"k": 10}).json()["recommendations"]
        matches += sum(r["product_type_name"] == favourite for r in recs)
    assert matches / (len(sample) * 10) > 0.5


def test_context_date_is_used(client, users):
    data = client.get(f"/recommend/{existing_customer(users)}", params={"date": "2020-12-24"}).json()
    assert data["context"] == {"month": 12, "day_of_week": "Thursday", "date": "2020-12-24"}


def test_exclude_purchased(client, users):
    customer_id = existing_customer(users)
    bought = {i["article_id"] for i in client.get(f"/customers/{customer_id}").json()["recent_purchases"]}
    recs = client.get(f"/recommend/{customer_id}", params={"exclude_purchased": True, "k": 20}).json()
    assert bought and not bought & {r["article_id"] for r in recs["recommendations"]}


def test_new_customer_gets_model_recommendations(client, users):
    customer_id = users[users["is_new_user"] == 1].iloc[0]["customer_id"]
    data = client.get(f"/recommend/{customer_id}", params={"k": 5}).json()
    assert data["source"] == "two_tower_model"
    assert data["customer"]["is_new_customer"] is True
    assert len(data["recommendations"]) == 5


def test_unknown_customer_gets_popular_fallback(client):
    data = client.get("/recommend/not-a-real-customer", params={"k": 12}).json()
    assert data["source"] == "popular_fallback"
    from src import config
    top12 = json.loads(Path(config.DATA_PATH, "fallback_top12.json").read_text())
    assert [r["article_id"] for r in data["recommendations"]] == top12


def test_similar_items(client):
    from src import config
    article_id = json.loads(Path(config.ITEM_IDS_FILE).read_text())[0]
    data = client.get(f"/similar/{article_id}", params={"k": 5}).json()
    assert len(data["similar"]) == 5
    assert article_id not in [i["article_id"] for i in data["similar"]]
    assert data["article"]["article_id"] == article_id


def test_customer_profile_and_history(client, users):
    data = client.get(f"/customers/{existing_customer(users)}").json()
    assert data["profile"]["n_purchases"] > 0
    assert 0 < len(data["recent_purchases"]) <= 12


def test_not_found_and_validation(client):
    assert client.get("/similar/0000000000").status_code == 404
    assert client.get("/customers/nobody").status_code == 404
    assert client.get("/recommend/x", params={"k": 0}).status_code == 422
    assert client.get("/recommend/x", params={"date": "not-a-date"}).status_code == 422


def test_sample_customers(client):
    data = client.get("/sample-customers").json()
    assert data["existing"] and data["new"] and data["examples"]


def test_frontend_page(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert "Recommend" in response.text


def test_stats_and_response_time_header(client, users):
    response = client.get(f"/recommend/{existing_customer(users)}")
    assert float(response.headers["X-Response-Time-ms"]) > 0
    assert client.get("/stats").json()["/recommend/{customer_id}"]["requests"] > 0


def test_admin_reload(client):
    assert client.post("/admin/reload").json()["status"] == "reloaded"
