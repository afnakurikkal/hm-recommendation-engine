"""Live API tests against the real model and the real feature store.

Start Redis + the API first (see README), then run:
    python -m pytest tests/test_api.py -v
The tests are skipped automatically when no API is running at API_URL.
"""
import os

import pytest
import requests

API_URL = os.environ.get("LIVE_API_URL") or os.environ.get("API_URL", "http://127.0.0.1:8000")


def _api_ready():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _api_ready(), reason=f"no running API with a loaded model at {API_URL}")


@pytest.fixture(scope="module")
def samples():
    return requests.get(f"{API_URL}/sample-customers", timeout=60).json()


def test_home():
    response = requests.get(f"{API_URL}/", timeout=30)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_health():
    data = requests.get(f"{API_URL}/health", timeout=30).json()
    assert data["redis"] is True
    assert data["items_in_index"] > 0


def test_recommend_existing_customer(samples):
    data = requests.get(f"{API_URL}/recommend/{samples['existing']}", params={"k": 12}, timeout=60).json()
    assert data["source"] == "two_tower_model"
    assert len(data["recommendations"]) == 12
    assert len({item["article_id"] for item in data["recommendations"]}) == 12   # no duplicates
    assert all(item["product_type_name"] for item in data["recommendations"])


def test_recommend_new_customer_uses_model_features(samples):
    if not samples.get("new"):
        pytest.skip("no customers without purchases loaded")
    data = requests.get(f"{API_URL}/recommend/{samples['new']}", params={"k": 5}, timeout=60).json()
    assert data["source"] == "two_tower_model"
    assert data["customer"]["is_new_customer"] is True
    assert len(data["recommendations"]) == 5


def test_context_changes_recommendations(samples):
    customer = samples["existing"]
    summer = requests.get(f"{API_URL}/recommend/{customer}", params={"date": "2020-07-04"}, timeout=60).json()
    winter = requests.get(f"{API_URL}/recommend/{customer}", params={"date": "2020-12-19"}, timeout=60).json()
    assert summer["context"]["month"] == 7 and winter["context"]["month"] == 12
    assert [i["article_id"] for i in summer["recommendations"]] != [i["article_id"] for i in winter["recommendations"]]


def test_unknown_customer_gets_popular_fallback():
    data = requests.get(f"{API_URL}/recommend/not-a-real-customer", params={"k": 12}, timeout=60).json()
    assert data["source"] == "popular_fallback"
    assert len(data["recommendations"]) == 12


def test_customer_history(samples):
    data = requests.get(f"{API_URL}/customers/{samples['existing']}", timeout=60).json()
    assert data["recent_purchases"]


def test_similar_items():
    article_id = requests.get(f"{API_URL}/recommend/x", timeout=60).json()["recommendations"][0]["article_id"]
    data = requests.get(f"{API_URL}/similar/{article_id}", params={"k": 5}, timeout=60).json()
    assert len(data["similar"]) == 5
    assert article_id not in [item["article_id"] for item in data["similar"]]


def test_model_metrics():
    data = requests.get(f"{API_URL}/model", timeout=30).json()
    assert data["test_metrics"]["two_tower_model"]["Recall@12"] > 0


def test_frontend_page():
    response = requests.get(f"{API_URL}/ui", timeout=30)
    assert response.status_code == 200
    assert "Recommend" in response.text


def test_bad_k_is_rejected():
    assert requests.get(f"{API_URL}/recommend/x", params={"k": 0}, timeout=30).status_code == 422
