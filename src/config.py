"""Paths and constants shared by every pipeline step and the API.

All paths can be changed with environment variables, so the same code runs in
Google Colab (default paths) or on another machine.
"""
import os

DATA_PATH = os.environ.get("HM_DATA_PATH", "/content/drive/MyDrive/hm_project")
MODEL_DIR = os.environ.get("HM_MODEL_DIR", os.path.join(DATA_PATH, "week2_model"))
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

CUTOFF = os.environ.get("HM_CUTOFF", "2020-09-16")          # train < cutoff <= test
# Date used as "today" when recommending (month / day-of-week context).
# The dataset ends in Sept 2020, so the default is the cutoff date.
CONTEXT_DATE = os.environ.get("HM_CONTEXT_DATE", CUTOFF)

ITEM_CAT_COLS = ["product_type_name", "colour_group_name", "department_name", "garment_group_name"]
USER_NUM_COLS = [
    "age", "age_missing", "log_n_purchases", "log_n_unique_items", "avg_price",
    "log_days_since_last_purchase", "is_new_user",
]
ITEM_NUM_COLS = [
    "log_purchases_last_7d", "log_purchases_last_28d", "log_purchases_all_time", "is_new_item",
]

ITEM_EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "item_embeddings.npy")
ITEM_IDS_FILE = os.path.join(MODEL_DIR, "item_ids.json")
MODEL_CONFIG_FILE = os.path.join(MODEL_DIR, "model_config.json")
TRAINED_USERS_FILE = os.path.join(MODEL_DIR, "trained_user_idx.npy")
QUERY_WEIGHTS_FILE = os.path.join(MODEL_DIR, "query_tower.weights.h5")
CANDIDATE_WEIGHTS_FILE = os.path.join(MODEL_DIR, "candidate_tower.weights.h5")
ANN_INDEX_FILE = os.path.join(MODEL_DIR, "item_hnsw.index")
