"""Run the whole offline pipeline with one command (the same steps as the Airflow weekly DAG).

    python scripts/run_pipeline.py --source path/to/h-and-m-personalized-fashion-recommendations.zip

Steps (choose with --steps, default: all):
  data   Week 1  raw CSVs -> features, vocabularies, train/test split   (scripts/build_local_data.py)
  train  Week 2  train + evaluate the two-tower model, export embeddings (src/training/train_two_tower.py)
  rank   Week 2  tune the ranking stage on the validation week          (src/training/tune_ranking.py)
  index  Week 3  build the FAISS HNSW index and benchmark it            (src/serving/ann_search.py)
  store  Week 3  load the Redis feature store                           (src/serving/load_redis.py)

Training settings come from environment variables (TRAIN_WEEKS, EPOCHS, BATCH_SIZE, LEARNING_RATE).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

STEPS = {
    "data": "Week 1 - feature pipeline",
    "train": "Week 2 - two-tower training + evaluation",
    "rank": "Week 2 - ranking stage (buy-again + popularity prior) tuned on the validation week",
    "index": "Week 3 - FAISS HNSW index",
    "store": "Week 3 - Redis feature store",
}


def run(title, args):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)
    started = time.time()
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env)
    if result.returncode != 0:
        sys.exit(f"\n{title} failed (exit code {result.returncode}).")
    print(f"-- {title}: done in {(time.time() - started) / 60:.1f} min", flush=True)


def redis_ok():
    try:
        from src.serving.store import connect
        connect()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Redis is not reachable at {config.REDIS_URL}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=os.environ.get("HM_RAW_DATA"),
                        help="Kaggle zip or folder with the CSVs (needed for the 'data' step)")
    parser.add_argument("--steps", default="data,train,rank,index,store", help="comma-separated: " + ",".join(STEPS))
    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {unknown}")

    print(f"data folder : {config.DATA_PATH}\nmodel folder: {config.MODEL_DIR}\nsteps       : {', '.join(steps)}")
    started = time.time()
    if "data" in steps:
        if not args.source:
            parser.error("the 'data' step needs --source (path to the Kaggle zip or CSV folder)")
        run(STEPS["data"], ["scripts/build_local_data.py", "--source", args.source])
    if "train" in steps:
        run(STEPS["train"], ["-m", "src.training.train_two_tower"])
    if "rank" in steps:
        run(STEPS["rank"], ["-m", "src.training.tune_ranking"])
    if "index" in steps:
        run(STEPS["index"], ["-m", "src.serving.ann_search"])
    if "store" in steps:
        from src.serving.store import is_memory
        if is_memory():
            print("\nREDIS_URL=memory:// - the API loads the feature store itself at start-up, skipping this step.")
        elif redis_ok():
            run(STEPS["store"], ["-m", "src.serving.load_redis"])
        else:
            sys.exit("Start Redis first (docker compose up -d redis) and run again with --steps store")

    print(f"\nAll done in {(time.time() - started) / 60:.1f} min. Start the API:\n"
          f"  uvicorn src.api.main:app --port 8000\nthen open http://localhost:8000")


if __name__ == "__main__":
    main()
