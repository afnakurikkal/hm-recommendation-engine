"""Week 3, Days 4-6: load item vectors and user profiles into Redis."""
import json
import os
import time

import numpy as np
import pandas as pd
import redis

DATA = "/content/drive/MyDrive/hm_project"
EXPORT = os.path.join(DATA, "export")
CUTOFF = pd.Timestamp("2020-09-16")
RECENT_FROM = "2020-08-19"          # last 28 days before the cutoff
ITEM_COLS = ["product_type_name", "colour_group_name", "department_name", "garment_group_name"]
BATCH = 10_000

r = redis.Redis(host="localhost", port=6379)   # raw bytes, vectors are stored as bytes


def load_items():
    emb = np.load(f"{EXPORT}/item_embeddings.npy").astype("float32")
    ids = json.load(open(f"{EXPORT}/item_ids.json"))
    articles = pd.read_csv(f"{DATA}/articles.csv", dtype={"article_id": str},
                           usecols=["article_id"] + ITEM_COLS)
    articles["article_id"] = articles["article_id"].str.zfill(10)
    meta = articles.set_index("article_id")[ITEM_COLS].to_dict("index")

    pipe = r.pipeline()
    for i, aid in enumerate(ids):
        mapping = {"vec": emb[i].tobytes()}
        for c in ITEM_COLS:
            mapping[c] = str(meta[aid][c])
        pipe.hset(f"item:{aid}", mapping=mapping)
        if (i + 1) % BATCH == 0:
            pipe.execute()
    pipe.execute()
    print("items loaded:", len(ids))


def build_user_stats():
    parts, pop = [], pd.Series(dtype="float64")
    reader = pd.read_csv(f"{DATA}/transactions_train.csv",
                         usecols=["t_dat", "customer_id", "article_id", "price"],
                         dtype={"article_id": str}, chunksize=2_000_000)
    for chunk in reader:
        chunk = chunk[chunk["t_dat"] < "2020-09-16"]
        parts.append(chunk.groupby("customer_id").agg(
            n=("price", "size"), price_sum=("price", "sum"), last=("t_dat", "max")))
        recent = chunk[chunk["t_dat"] >= RECENT_FROM]
        pop = pop.add(recent["article_id"].str.zfill(10).value_counts(), fill_value=0)

    g = pd.concat(parts).groupby(level=0).agg(
        n=("n", "sum"), price_sum=("price_sum", "sum"), last=("last", "max"))
    stats = pd.DataFrame({
        "n_purchases": g["n"],
        "avg_price": g["price_sum"] / g["n"],
        "days_since_last_purchase": (CUTOFF - pd.to_datetime(g["last"])).dt.days,
    })
    return stats, pop.sort_values(ascending=False)


def load_users(stats):
    c = pd.read_csv(f"{DATA}/customers.csv",
                    usecols=["customer_id", "club_member_status", "fashion_news_frequency", "age"])
    c["age"] = c["age"].fillna(c["age"].median())
    c["club_status"] = c["club_member_status"].fillna("UNKNOWN")
    c["news_frequency"] = c["fashion_news_frequency"].fillna("NONE").str.upper()
    c = c.merge(stats, left_on="customer_id", right_index=True, how="left")
    c["n_purchases"] = c["n_purchases"].fillna(0)
    c["avg_price"] = c["avg_price"].fillna(0.0)
    c["days_since_last_purchase"] = c["days_since_last_purchase"].fillna(-1)   # -1 = no purchases

    pipe = r.pipeline()
    for i, row in enumerate(c.itertuples(index=False)):
        pipe.hset(f"user:{row.customer_id}", mapping={
            "age": int(row.age),
            "club_status": row.club_status,
            "news_frequency": row.news_frequency,
            "n_purchases": int(row.n_purchases),
            "avg_price": round(float(row.avg_price), 5),
            "days_since_last_purchase": int(row.days_since_last_purchase),
        })
        if (i + 1) % BATCH == 0:
            pipe.execute()
    pipe.execute()
    print("users loaded:", len(c))
    return c["customer_id"].iloc[0]


def smoke_test(user_id, item_id):
    print("user profile:", r.hgetall(f"user:{user_id}"))
    vec = np.frombuffer(r.hget(f"item:{item_id}", "vec"), dtype="float32")
    print("item vector shape:", vec.shape)
    start = time.perf_counter()
    for _ in range(1000):
        r.hgetall(f"user:{user_id}")
        r.hget(f"item:{item_id}", "vec")
    ms = (time.perf_counter() - start) / 1000 * 1000
    print(f"avg lookup (user + item vector): {ms:.3f} ms")


def main():
    print("redis:", r.ping())
    t = time.time()
    load_items()
    stats, pop = build_user_stats()
    r.set("fallback:top12", ",".join(pop.index[:12]))
    r.set("fallback:top100", ",".join(pop.index[:100]))
    first_user = load_users(stats)
    first_item = json.load(open(f"{EXPORT}/item_ids.json"))[0]
    smoke_test(first_user, first_item)
    print("keys in Redis:", r.dbsize(), "| memory:", r.info("memory")["used_memory_human"])
    print(f"done in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()