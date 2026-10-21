"""Week 4, Days 6-7: stress test the API under concurrent load.

Sends many /recommend requests in parallel (real customers from the feature
store, plus some unknown ones) and reports throughput and latency percentiles.

Run (API + Redis must be running):  python -m tests.stress_test
Options: API_URL, STRESS_LEVELS (e.g. "1,10,50"), STRESS_REQUESTS (per level).
"""
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import redis
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
LEVELS = [int(x) for x in os.environ.get("STRESS_LEVELS", "1,10,25,50").split(",")]
REQUESTS_PER_LEVEL = int(os.environ.get("STRESS_REQUESTS", 500))


def sample_customers(n=2000):
    """Real customers from the Redis feature store (or the API's examples) + ~10% unknown ones."""
    ids = []
    try:
        r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=3)
        for key in r.scan_iter("user:*", count=5000):
            ids.append(key.decode().split(":", 1)[1])
            if len(ids) >= n:
                break
    except redis.RedisError:
        samples = requests.get(f"{API_URL}/sample-customers", timeout=60).json()
        ids = [samples["existing"], *samples.get("examples", [])] + ([samples["new"]] if samples.get("new") else [])
    return ids + [f"unknown-customer-{i}" for i in range(max(1, len(ids) // 10))]   # ~10% cold-start


def run_level(concurrency, customers):
    local = threading.local()

    def call(customer_id):
        if not hasattr(local, "session"):
            local.session = requests.Session()        # one connection pool per thread
        session = local.session
        start = time.perf_counter()
        try:
            ok = session.get(f"{API_URL}/recommend/{customer_id}", params={"k": 12}, timeout=30).status_code == 200
        except requests.RequestException:
            ok = False
        return ok, (time.perf_counter() - start) * 1000

    picks = [random.choice(customers) for _ in range(REQUESTS_PER_LEVEL)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(call, picks))
    elapsed = time.perf_counter() - started
    latencies = np.array([ms for ok, ms in results if ok])
    errors = sum(1 for ok, _ in results if not ok)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": errors,
        "throughput_rps": len(results) / elapsed,
        "p50_ms": float(np.percentile(latencies, 50)) if len(latencies) else float("nan"),
        "p95_ms": float(np.percentile(latencies, 95)) if len(latencies) else float("nan"),
        "p99_ms": float(np.percentile(latencies, 99)) if len(latencies) else float("nan"),
    }


def main():
    customers = sample_customers()
    print(f"customers sampled: {len(customers)} | requests per level: {REQUESTS_PER_LEVEL}\n")
    print(f"{'users':>6} {'requests':>9} {'errors':>7} {'req/s':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}")
    rows = []
    for level in LEVELS:
        row = run_level(level, customers)
        rows.append(row)
        print(f"{row['concurrency']:>6} {row['requests']:>9} {row['errors']:>7} {row['throughput_rps']:>8.1f} "
              f"{row['p50_ms']:>8.1f} {row['p95_ms']:>8.1f} {row['p99_ms']:>8.1f}")
    out = os.environ.get("STRESS_OUTPUT", os.path.join("artifacts", "stress_test_results.json"))
    if os.path.isdir(os.path.dirname(out) or "."):
        with open(out, "w") as f:
            json.dump({"api_url": API_URL, "run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "levels": rows}, f, indent=2)
        print("\nresults saved to", out)
    return rows


if __name__ == "__main__":
    main()
