"""Write the latest load-test numbers from artifacts/stress_test_results.json into README.md.

    python scripts/update_readme_results.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def main():
    stress = json.loads((ROOT / "artifacts" / "stress_test_results.json").read_text())
    lines = ["**API under load** (`tests/stress_test.py`: real customers + 10% unknown visitors, Redis + "
             "TensorFlow + FAISS, one uvicorn worker on a laptop):", "",
             "| Concurrent users | Requests | Errors | Throughput | p50 | p95 | p99 |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['concurrency']} | {r['requests']} | {r['errors']} | {r['throughput_rps']:.0f} req/s | "
              f"{r['p50_ms']:.1f} ms | {r['p95_ms']:.1f} ms | {r['p99_ms']:.1f} ms |" for r in stress["levels"]]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    (ROOT / "README.md").write_text(f"{head}{START}\n" + "\n".join(lines) + f"\n{END}{tail}", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
