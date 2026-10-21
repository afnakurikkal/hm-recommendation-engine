"""Paths and constants shared by every pipeline step and the API.

All paths can be changed with environment variables, so the same code runs in
Google Colab (Google Drive paths), on a laptop (``data/hm_project`` inside the
repo) or in Docker.
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
_COLAB_DATA = "/content/drive/MyDrive/hm_project"
_LOCAL_DATA = str(PROJECT_DIR / "data" / "hm_project")

DATA_PATH = os.environ.get("HM_DATA_PATH") or (_COLAB_DATA if os.path.isdir(_COLAB_DATA) else _LOCAL_DATA)
MODEL_DIR = os.environ.get("HM_MODEL_DIR", os.path.join(DATA_PATH, "week2_model"))
ARTIFACTS_DIR = os.environ.get("HM_ARTIFACTS_DIR", str(PROJECT_DIR / "artifacts"))

# Redis feature store. REDIS_URL wins over host/port. "memory://" = in-process
# store (no Redis server needed; the API loads the features itself at start-up).
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_URL = os.environ.get("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

# Product photos: the Kaggle zip or a folder that contains images/. Optional -
# the web UI shows a colour swatch when a photo is not available.
IMAGES_SOURCE = os.environ.get("HM_IMAGES")

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
# Extra article fields shown in the web UI (from articles.csv; optional).
ITEM_DETAIL_COLS = ["prod_name", "product_group_name", "graphical_appearance_name",
                    "index_name", "section_name", "detail_desc"]

ITEM_EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "item_embeddings.npy")
ITEM_IDS_FILE = os.path.join(MODEL_DIR, "item_ids.json")
MODEL_CONFIG_FILE = os.path.join(MODEL_DIR, "model_config.json")
TRAINED_USERS_FILE = os.path.join(MODEL_DIR, "trained_user_idx.npy")
QUERY_WEIGHTS_FILE = os.path.join(MODEL_DIR, "query_tower.weights.h5")
CANDIDATE_WEIGHTS_FILE = os.path.join(MODEL_DIR, "candidate_tower.weights.h5")
ANN_INDEX_FILE = os.path.join(MODEL_DIR, "item_hnsw.index")
