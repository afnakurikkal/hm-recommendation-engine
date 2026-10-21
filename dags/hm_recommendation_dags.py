"""Week 4, Days 4-5: Apache Airflow DAGs for the H&M recommendation pipeline.

hm_weekly_retraining (every Sunday 02:00)
    build_features -> train_two_tower -> tune_ranking -> build_ann_index -> load_feature_store -> reload_api -> api_smoke_test

hm_daily_embedding_refresh (every day 03:00)
    build_features -> refresh_item_embeddings -> build_ann_index -> load_feature_store -> reload_api

Set the Airflow Variable / env var HM_PROJECT_DIR to the repo folder and
HM_DATA_PATH to the data folder (default: the Google Drive folder used in Colab).
"""
import os
from datetime import datetime, timedelta

from airflow import DAG

try:                                   # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:                    # Airflow 2
    from airflow.operators.bash import BashOperator

PROJECT_DIR = os.environ.get("HM_PROJECT_DIR", "/opt/hm-recommendation-engine")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
ENV = {
    "HM_DATA_PATH": os.environ.get("HM_DATA_PATH", "/content/drive/MyDrive/hm_project"),
    "PYTHONPATH": PROJECT_DIR,
    "PATH": os.environ.get("PATH", ""),
}


def step(module):
    return f"cd {PROJECT_DIR} && python -m {module}"


default_args = {
    "owner": "hm-recsys",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="hm_weekly_retraining",
    description="Weekly: rebuild features, retrain the two-tower model, rebuild ANN index and Redis",
    default_args=default_args,
    start_date=datetime(2026, 11, 1),
    schedule="0 2 * * 0",
    catchup=False,
    max_active_runs=1,
    tags=["hm", "recsys", "training"],
) as weekly:
    build_features = BashOperator(task_id="build_features", bash_command=step("src.pipeline.build_features"),
                                  env=ENV, execution_timeout=timedelta(hours=2))
    train_model = BashOperator(task_id="train_two_tower", bash_command=step("src.training.train_two_tower"),
                               env=ENV, execution_timeout=timedelta(hours=3))
    tune_ranking = BashOperator(task_id="tune_ranking", bash_command=step("src.training.tune_ranking"),
                                env=ENV, execution_timeout=timedelta(hours=1))
    build_index = BashOperator(task_id="build_ann_index", bash_command=step("src.serving.ann_search"), env=ENV)
    load_store = BashOperator(task_id="load_feature_store", bash_command=step("src.serving.load_redis"), env=ENV)
    reload_api = BashOperator(task_id="reload_api", bash_command=f"curl -sf -X POST {API_URL}/admin/reload")
    smoke_test = BashOperator(task_id="api_smoke_test", bash_command=f"curl -sf {API_URL}/health")

    build_features >> train_model >> tune_ranking >> build_index >> load_store >> reload_api >> smoke_test

with DAG(
    dag_id="hm_daily_embedding_refresh",
    description="Daily: refresh features and item embeddings with the current model (no retraining)",
    default_args=default_args,
    start_date=datetime(2026, 11, 1),
    schedule="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["hm", "recsys", "embeddings"],
) as daily:
    refresh_features = BashOperator(task_id="build_features", bash_command=step("src.pipeline.build_features"),
                                    env=ENV, execution_timeout=timedelta(hours=2))
    refresh_embeddings = BashOperator(task_id="refresh_item_embeddings",
                                      bash_command=step("src.serving.export_embeddings"), env=ENV)
    rebuild_index = BashOperator(task_id="build_ann_index", bash_command=step("src.serving.ann_search"), env=ENV)
    reload_store = BashOperator(task_id="load_feature_store", bash_command=step("src.serving.load_redis"), env=ENV)
    reload_service = BashOperator(task_id="reload_api", bash_command=f"curl -sf -X POST {API_URL}/admin/reload")

    refresh_features >> refresh_embeddings >> rebuild_index >> reload_store >> reload_service
