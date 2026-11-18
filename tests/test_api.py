"""API tests. Start the API and Redis first, then run:  python -m pytest tests/test_api.py -v"""
import os

import redis
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
r = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), port=int(os.environ.get("REDIS_PORT", 6379)))


def find_customer(new_customer):
    """A customer from the feature store: with (False) or without (True) purchase history."""
    for key in r.scan_iter("user:*", count=1000):
        if (r.hget(key, "is_new_user") == b"1") == new_customer:
            return key.decode().split(":", 1)[1]
    return None


def test_home():
    response = requests.get(f"{API_URL}/", timeout=30)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_health():
    data = requests.get(f"{API_URL}/health", timeout=30).json()
    assert data["redis"] is True
    assert data["items_in_index"] > 0


def test_recommend_existing_customer():
    customer_id = find_customer(new_customer=False)
    data = requests.get(f"{API_URL}/recommend/{customer_id}", params={"k": 12}, timeout=60).json()
    assert data["source"] == "two_tower_model"
    assert len(data["recommendations"]) == 12
    assert len({item["article_id"] for item in data["recommendations"]}) == 12   # no duplicates
    assert all(item["product_type_name"] for item in data["recommendations"])


def test_recommend_new_customer_uses_model_features():
    customer_id = find_customer(new_customer=True)
    if customer_id is None:
        return  # no new customers loaded (e.g. HM_MAX_USERS demo mode)
    data = requests.get(f"{API_URL}/recommend/{customer_id}", params={"k": 5}, timeout=60).json()
    assert data["source"] == "two_tower_model"
    assert data["customer"]["is_new_customer"] is True
    assert len(data["recommendations"]) == 5


def test_unknown_customer_gets_popular_fallback():
    data = requests.get(f"{API_URL}/recommend/not-a-real-customer", params={"k": 12}, timeout=60).json()
    assert data["source"] == "popular_fallback"
    assert len(data["recommendations"]) == 12


def test_similar_items():
    article_id = r.scan(cursor=0, match="item:*", count=1000)[1][0].decode().split(":", 1)[1]
    data = requests.get(f"{API_URL}/similar/{article_id}", params={"k": 5}, timeout=60).json()
    assert len(data["similar"]) == 5
    assert article_id not in [item["article_id"] for item in data["similar"]]


def test_bad_k_is_rejected():
    assert requests.get(f"{API_URL}/recommend/x", params={"k": 0}, timeout=30).status_code == 422
