"""Week 3, Days 1-3: train the two-tower model on a sample, then export
the towers and the item embeddings."""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.candidate_tower import CandidateTower
from src.models.query_tower import QueryTower

DATA = "/content/drive/MyDrive/hm_project"
OUT = os.path.join(DATA, "export")
CUTOFF = "2020-09-16"
ITEM_COLS = ["product_type_name", "colour_group_name", "department_name", "garment_group_name"]
EMB_DIM = 32
BATCH = 1024
TEMP = 0.05
EPOCHS = int(os.environ.get("EPOCHS", 3))
FRAC = float(os.environ.get("SAMPLE_FRAC", 0.05))


def pad_id(s):
    return s.astype(str).str.zfill(10)


def load_data():
    articles = pd.read_csv(f"{DATA}/articles.csv", dtype={"article_id": str})
    articles["article_id"] = pad_id(articles["article_id"])
    for c in ITEM_COLS:
        articles[c] = articles[c].astype(str)

    customers = pd.read_csv(
        f"{DATA}/customers.csv",
        usecols=["customer_id", "club_member_status", "fashion_news_frequency", "age"],
    )
    customers["age"] = customers["age"].fillna(customers["age"].median()).astype("float32")
    customers["club_member_status"] = customers["club_member_status"].fillna("UNKNOWN")
    customers["fashion_news_frequency"] = (
        customers["fashion_news_frequency"].fillna("NONE").str.upper()
    )
    customers = customers.rename(
        columns={"club_member_status": "club_status", "fashion_news_frequency": "news_frequency"}
    )

    train_parts, test_parts = [], []
    reader = pd.read_csv(
        f"{DATA}/transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"article_id": str},
        chunksize=1_000_000,
    )
    for chunk in reader:
        train_parts.append(chunk[chunk["t_dat"] < CUTOFF].sample(frac=FRAC, random_state=42))
        test_parts.append(chunk[chunk["t_dat"] >= CUTOFF])
    train, test = pd.concat(train_parts), pd.concat(test_parts)
    train["article_id"] = pad_id(train["article_id"])
    test["article_id"] = pad_id(test["article_id"])

    keep = ["article_id"] + ITEM_COLS
    train = train.merge(customers, on="customer_id").merge(articles[keep], on="article_id")
    test = test.merge(customers, on="customer_id")
    return articles, train, test


def to_ds(df):
    cols = ["customer_id", "club_status", "news_frequency", "article_id"] + ITEM_COLS
    data = {c: df[c].to_numpy(dtype=str) for c in cols}
    data["age"] = df["age"].to_numpy(dtype="float32")
    ds = tf.data.Dataset.from_tensor_slices(data)
    return ds.shuffle(200_000).batch(BATCH, drop_remainder=True).prefetch(2)


def query_inputs(b):
    return {k: b[k] for k in ["customer_id", "club_status", "news_frequency", "age"]}


def cand_inputs(b):
    return {k: b[k] for k in ["article_id"] + ITEM_COLS}


def embed_items(cand_tower, articles):
    out = []
    for i in range(0, len(articles), 4096):
        part = articles.iloc[i : i + 4096]
        inputs = {c: tf.constant(part[c].to_numpy(dtype=str)) for c in ["article_id"] + ITEM_COLS}
        out.append(tf.math.l2_normalize(cand_tower(inputs), axis=1).numpy())
    return np.vstack(out).astype("float32")


def hit_rate_at_10(query_tower, item_emb, article_ids, test, n=5000):
    test = test.sample(min(n, len(test)), random_state=1)
    row_of = {a: i for i, a in enumerate(article_ids)}
    hits = 0
    for i in range(0, len(test), 500):
        part = test.iloc[i : i + 500]
        inputs = {c: tf.constant(part[c].to_numpy(dtype=str))
                  for c in ["customer_id", "club_status", "news_frequency"]}
        inputs["age"] = tf.constant(part["age"].to_numpy(dtype="float32"))
        q = tf.math.l2_normalize(query_tower(inputs), axis=1).numpy()
        top = np.argsort(-(q @ item_emb.T), axis=1)[:, :10]
        target = part["article_id"].map(row_of).to_numpy()
        hits += int((top == target[:, None]).any(axis=1).sum())
    return hits / len(test)


def main():
    os.makedirs(OUT, exist_ok=True)
    articles, train, test = load_data()
    print("train rows:", len(train), "| test rows:", len(test))

    customer_vocab = train["customer_id"].unique().tolist()
    club_vocab = sorted(train["club_status"].unique().tolist())
    news_vocab = sorted(train["news_frequency"].unique().tolist())
    article_vocab = articles["article_id"].tolist()
    metadata_vocabs = {c: sorted(articles[c].unique().tolist()) for c in ITEM_COLS}

    query_tower = QueryTower(customer_vocab, club_vocab, news_vocab, embedding_dim=EMB_DIM)
    cand_tower = CandidateTower(article_vocab, metadata_vocabs, embedding_dim=EMB_DIM)
    ds = to_ds(train)

    first = next(iter(ds))              # build the weights before training
    query_tower(query_inputs(first))
    cand_tower(cand_inputs(first))
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def train_step(b):
        with tf.GradientTape() as tape:
            q = tf.math.l2_normalize(query_tower(query_inputs(b)), axis=1)
            c = tf.math.l2_normalize(cand_tower(cand_inputs(b), training=True), axis=1)
            logits = tf.matmul(q, c, transpose_b=True) / TEMP
            idx = cand_tower.article_lookup(b["article_id"])
            pos = tf.cast(tf.equal(idx[:, None], idx[None, :]), tf.float32)
            pos_logits = logits + (1.0 - pos) * -1e9      # multi-positive loss
            loss = tf.reduce_mean(
                tf.reduce_logsumexp(logits, axis=1) - tf.reduce_logsumexp(pos_logits, axis=1)
            )
        variables = query_tower.trainable_variables + cand_tower.trainable_variables
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    for epoch in range(EPOCHS):
        losses = []
        for step, batch in enumerate(ds):
            losses.append(float(train_step(batch)))
            if step % 200 == 0:
                print(f"epoch {epoch + 1} step {step} loss {np.mean(losses[-200:]):.4f}")
        print(f"epoch {epoch + 1} mean loss {np.mean(losses):.4f}")

    item_emb = embed_items(cand_tower, articles)
    print("item embeddings:", item_emb.shape)
    print("Hit rate@10 on test period:", hit_rate_at_10(query_tower, item_emb, article_vocab, test))

    np.save(f"{OUT}/item_embeddings.npy", item_emb)
    json.dump(article_vocab, open(f"{OUT}/item_ids.json", "w"))
    query_tower.save_weights(f"{OUT}/query_tower.weights.h5")
    cand_tower.save_weights(f"{OUT}/candidate_tower.weights.h5")
    json.dump(
        {"customer_vocab": customer_vocab, "club_vocab": club_vocab, "news_vocab": news_vocab,
         "metadata_vocabs": metadata_vocabs, "embedding_dim": EMB_DIM, "item_cols": ITEM_COLS},
        open(f"{OUT}/model_config.json", "w"),
    )
    print("Saved to", OUT)


if __name__ == "__main__":
    main()