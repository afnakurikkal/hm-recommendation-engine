"""Ranking stage: tune how the final top-12 is built (on the validation week only).

Two candidate sources are combined:

1. Repurchase: articles the customer bought in the last D days (newest first).
   Re-buying (other size / colour of the same article) is very common at H&M.
2. Two-tower model, ranked by
       cosine(user, item) / temperature + w * log P_recent(item)
   The model is trained with logQ correction, which removes popularity bias on
   purpose; w adds the recent-popularity prior back at serving time.

Final list = repurchase candidates first, then the model's top items.

w and D are chosen on the VALIDATION week (the week before the cutoff; history
and popularity taken from before that week). The test week is used only once,
for the final numbers, which are reported for every variant - including
"repurchase + popularity" without the model, to show what the model adds.

Run after training:  python -m src.training.tune_ranking
Writes popularity_weight, repurchase_days and the test metrics to
model_config.json and artifacts/.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from src import config
from src.data.features import apply_standardizer, context_features, encode, load_vocabularies, user_numeric_frame
from src.models.loader import load_model_config, load_query_tower, load_trained_users
from src.training.train_two_tower import write_report

TOP_K = 12
POOL = 36                                    # model candidates kept per user before merging
WEIGHTS = [0.0, 0.5, 1.0, 2.0]
REPURCHASE_DAYS = [0, 7, 14, 28, 56]
VAL_USERS = 25_000


def log_prior(counts):
    counts = counts.astype("float64")
    return np.log((counts + 1) / (counts.sum() + len(counts))).astype("float32")


def evaluate(recs, users, truth):
    tot = {"Recall@12": 0.0, "NDCG@12": 0.0, "MAP@12": 0.0}
    discounts = 1 / np.log2(np.arange(2, TOP_K + 2))
    for user, rec in zip(users, recs):
        rel = truth[user]
        rec = list(rec)[:TOP_K]
        hits = np.zeros(TOP_K)
        hits[:len(rec)] = [x in rel for x in rec]
        n_rel = min(len(rel), TOP_K)
        tot["Recall@12"] += hits.sum() / len(rel)
        tot["NDCG@12"] += (hits * discounts).sum() / discounts[:n_rel].sum()
        tot["MAP@12"] += (np.cumsum(hits) / np.arange(1, TOP_K + 1) * hits).sum() / n_rel
    return {k: float(v / len(users)) for k, v in tot.items()}


def merge(first, rest):
    out = list(first[:TOP_K])
    seen = set(out)
    for item in rest:
        if len(out) >= TOP_K:
            break
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def repurchase_lists(history, users, days):
    """users -> articles bought within `days` (newest first); history has user_idx, item_idx, age (days)."""
    if days == 0:
        return [[] for _ in users]
    recent = history[history["age"] <= days].sort_values("age")
    recent = recent.drop_duplicates(["user_idx", "item_idx"])
    lists = recent.groupby("user_idx")["item_idx"].apply(list)
    return [lists.get(u, []) for u in users]


def main():
    started = time.time()
    cfg = load_model_config()
    temperature = cfg["temperature"]
    n_users, n_items = cfg["n_users"], cfg["n_items"]
    vocab = load_vocabularies()
    data = config.DATA_PATH

    # ---------------- customer query features (same encoding as training / serving)
    users = pd.read_parquet(os.path.join(data, "user_table.parquet"))
    u = users["user_idx"].to_numpy()
    trained = np.zeros(n_users + 1, bool)
    trained[list(load_trained_users())] = True
    club = np.zeros(n_users + 1, "int32"); club[u] = encode(users["club_member_status"], vocab["club_member_status"])
    news = np.zeros(n_users + 1, "int32"); news[u] = encode(users["fashion_news_frequency"], vocab["fashion_news_frequency"])
    num = np.zeros((n_users + 1, len(config.USER_NUM_COLS)), "float32")
    num[u] = apply_standardizer(user_numeric_frame(users), cfg["user_num_mean"], cfg["user_num_std"])
    tower = load_query_tower(cfg)
    catalog_t = tf.constant(np.load(config.ITEM_EMBEDDINGS_FILE).astype("float32").T)   # column c = item_idx c+1

    def model_pool(uidx, date, prior, weight):
        month, dow = context_features(date)
        out, prior_t = [], tf.constant(prior[None, :])
        for s in range(0, len(uidx), 512):
            b = uidx[s:s + 512]
            q = tower({"user_idx": tf.constant(np.where(trained[b], b, 0).astype("int32")),
                       "club": tf.constant(club[b]), "news": tf.constant(news[b]),
                       "month": tf.constant(np.full(len(b), month, "int32")),
                       "day_of_week": tf.constant(np.full(len(b), dow, "int32")),
                       "user_num": tf.constant(num[b])}, training=False)
            scores = tf.matmul(q, catalog_t) / temperature + weight * prior_t
            out.append(tf.math.top_k(scores, k=POOL).indices.numpy() + 1)
        return np.concatenate(out)

    inter = pd.read_parquet(os.path.join(data, "interactions_train.parquet"),
                            columns=["user_idx", "item_idx", "days_ago"])
    inter = inter[(inter["user_idx"] > 0) & (inter["item_idx"] > 0)]

    # ---------------- validation week = days 1-7 before the cutoff; everything else from before it
    val = inter[inter["days_ago"] <= 7]
    before_val = inter[inter["days_ago"] > 7].assign(age=lambda d: d["days_ago"] - 7)
    val_popular = np.bincount(before_val.loc[before_val["age"] <= 7, "item_idx"], minlength=n_items + 1)[1:]
    val_prior = log_prior(val_popular)
    val_top12 = (np.argsort(-val_popular, kind="stable")[:TOP_K] + 1).tolist()
    val_truth = val.groupby("user_idx")["item_idx"].apply(set)
    rng = np.random.default_rng(0)
    val_users = rng.choice(val_truth.index.to_numpy(), min(VAL_USERS, len(val_truth)), replace=False).astype("int32")
    val_date = (pd.Timestamp(config.CUTOFF) - pd.Timedelta(days=7)).date().isoformat()

    print(f"tuning on {len(val_users):,} validation-week customers")
    pools = {w: model_pool(val_users, val_date, val_prior, w) for w in WEIGHTS}
    rebuy = {d: repurchase_lists(before_val, val_users, d) for d in REPURCHASE_DAYS}
    tuning = {}
    for w in WEIGHTS:
        for d in REPURCHASE_DAYS:
            recs = [merge(r, p) for r, p in zip(rebuy[d], pools[w])]
            tuning[(w, d)] = evaluate(recs, val_users, val_truth)
            print(f"  w={w:<4} repurchase {d:>2} days | validation MAP@12 {tuning[(w, d)]['MAP@12']:.4f} "
                  f"Recall@12 {tuning[(w, d)]['Recall@12']:.4f}", flush=True)
    best_w, best_d = max(tuning, key=lambda key: tuning[key]["MAP@12"])
    print(f"chosen: popularity weight {best_w}, repurchase window {best_d} days")
    val_baseline = evaluate([merge(r, val_top12) for r in rebuy[best_d]], val_users, val_truth)

    # ---------------- test week: history / popularity up to the cutoff (exactly what serving uses)
    items = pd.read_parquet(os.path.join(data, "item_table.parquet"), columns=["item_idx", "purchases_last_7d"])
    counts = np.zeros(n_items + 1)
    counts[items["item_idx"]] = items["purchases_last_7d"]
    test_prior = log_prior(counts[1:])
    vocab_items = pd.read_parquet(os.path.join(data, "item_table.parquet"), columns=["article_id", "item_idx"])
    a2i = dict(zip(vocab_items["article_id"], vocab_items["item_idx"]))
    with open(os.path.join(data, "fallback_top12.json")) as f:
        top12 = [a2i[a] for a in json.load(f) if a in a2i]
    test = pd.read_parquet(os.path.join(data, "interactions_test.parquet"), columns=["user_idx", "item_idx"])
    test = test[(test["user_idx"] > 0) & (test["item_idx"] > 0)]
    truth = test.groupby("user_idx")["item_idx"].apply(set)
    test_users = truth.index.to_numpy().astype("int32")
    history = inter.assign(age=inter["days_ago"])
    test_rebuy = repurchase_lists(history, test_users, best_d)

    served_pool = model_pool(test_users, config.CUTOFF, test_prior, best_w)
    tower_pool = served_pool if best_w == 0 else model_pool(test_users, config.CUTOFF, test_prior, 0.0)
    results = {
        "two_tower_model": evaluate([merge(r, p) for r, p in zip(test_rebuy, served_pool)], test_users, truth),
        "two_tower_retrieval_only": evaluate(tower_pool, test_users, truth),
        "repurchase_plus_popularity": evaluate([merge(r, top12) for r in test_rebuy], test_users, truth),
        "popularity_baseline": evaluate([top12] * len(test_users), test_users, truth),
    }
    for name, m in results.items():
        print(f"{name:<28}", {k: round(v, 5) for k, v in m.items()})

    cfg["popularity_weight"] = best_w
    cfg["repurchase_days"] = best_d
    cfg["ranking_tuning"] = {f"w={w},days={d}": m for (w, d), m in tuning.items()}
    cfg["validation_repurchase_plus_popularity"] = val_baseline
    cfg["test_metrics"] = results
    with open(config.MODEL_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    write_report(cfg, data_note=(
        f"\n**Served ranking** = articles the customer bought in the last {best_d} days (newest first), then the "
        f"two-tower top items ranked by cosine / T + {best_w} x log(7-day popularity). Both settings were chosen on "
        "the validation week (`src/training/tune_ranking.py`); the test week was used once, for the numbers below.\n"))
    print(f"done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
