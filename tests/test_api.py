import pandas as pd
import requests

API_URL = "http://127.0.0.1:8002"

TRANSACTIONS_PATH = (
    "/content/drive/MyDrive/hm_project/transactions_train.csv"
)


def test_home_endpoint():
    response = requests.get(
        f"{API_URL}/",
        timeout=30
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"

    print("Home endpoint test passed!")


def test_recommendation_endpoint():
    transactions = pd.read_csv(
        TRANSACTIONS_PATH,
        dtype={"customer_id": str}
    )

    customer_id = transactions["customer_id"].iloc[0]

    response = requests.get(
        f"{API_URL}/recommend/{customer_id}",
        params={"k": 10},
        timeout=120
    )

    assert response.status_code == 200

    data = response.json()

    assert data["customer_id"] == customer_id
    assert data["requested_k"] == 10
    assert data["total_recommendations"] == 10
    assert len(data["recommendations"]) == 10

    print("Recommendation endpoint test passed!")
    print("Total recommendations:", data["total_recommendations"])


if __name__ == "__main__":
    test_home_endpoint()
    test_recommendation_endpoint()
    print("All API tests passed!")
