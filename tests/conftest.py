"""Test settings.

tests/test_pipeline.py runs the whole system offline on a small synthetic
dataset: these environment variables must be set before `src` is imported, so
the tests never touch the real data, model or Redis.
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="hm_tests_")
os.environ.update({
    "HM_DATA_PATH": os.path.join(_TMP, "data"),
    "HM_ARTIFACTS_DIR": os.path.join(_TMP, "artifacts"),
    "REDIS_URL": "memory://",
    "HM_MAX_USERS": "0",
    "HM_IMAGES": "",
    "TRAIN_WEEKS": "12",
    "EPOCHS": "6",
    "BATCH_SIZE": "64",
    "LEARNING_RATE": "0.005",
})
