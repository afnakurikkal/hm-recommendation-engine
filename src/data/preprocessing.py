
import os
import pandas as pd

DATA_PATH = "/content/drive/MyDrive/hm_project"


def load_reference_data(data_path=DATA_PATH):
    articles = pd.read_csv(
        os.path.join(data_path, "articles.csv"),
        dtype={"article_id": str}
    )

    customers = pd.read_csv(
        os.path.join(data_path, "customers.csv"),
        dtype={"customer_id": str}
    )

    return articles, customers


def validate_reference_data(articles, customers):
    assert articles["article_id"].is_unique
    assert customers["customer_id"].is_unique

    print("Articles:", len(articles))
    print("Customers:", len(customers))
    print("Reference data validation successful!")


def load_transaction_chunks(data_path=DATA_PATH, chunksize=500000):
    file_path = os.path.join(data_path, "transactions_train.csv")

    return pd.read_csv(
        file_path,
        chunksize=chunksize,
        dtype={"customer_id": str, "article_id": str}
    )
