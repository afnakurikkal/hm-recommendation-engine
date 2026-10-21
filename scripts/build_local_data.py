"""Week 1 feature pipeline - local edition (DuckDB instead of a Spark cluster).

Produces exactly the same files as `src/pipeline/build_features.py` (the PySpark
job used in Colab / Airflow), so training, the feature store and the API work the
same way. It runs on a normal laptop (Windows, macOS or Linux) and reads the
Kaggle download directly - the 30 GB zip does not need to be extracted.

    python scripts/build_local_data.py --source path/to/h-and-m-personalized-fashion-recommendations.zip

`--source` can also be a folder containing articles.csv, customers.csv and
transactions_train.csv. Output goes to HM_DATA_PATH (default: data/hm_project).

Extra files (used by the API / web UI, not by the model):
  article_details.parquet  product name, description, group, section ...
  user_history.parquet     each customer's most recent purchases (shown in the UI)
  build_info.json          where the raw data and product images are
"""
import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

ITEM_CAT_COLS = config.ITEM_CAT_COLS
DETAIL_COLS = ["prod_name", "product_group_name", "graphical_appearance_name",
               "perceived_colour_master_name", "index_name", "index_group_name",
               "section_name", "detail_desc"]


class RawData:
    """The three Kaggle CSVs, from the original zip or from an extracted folder."""

    def __init__(self, source):
        self.source = Path(source)
        self.zip = zipfile.ZipFile(self.source) if self.source.is_file() else None

    def open(self, name):
        if self.zip:
            return self.zip.open(name)
        return open(self.source / name, "rb")

    def has_images(self):
        if self.zip:
            return any(n.startswith("images/") for n in self.zip.namelist()[:50])
        return (self.source / "images").is_dir()


def csv_to_parquet(raw, name, out_file, column_types):
    """Stream one CSV into parquet in blocks (never loads the whole file in memory)."""
    if out_file.exists():
        print(f"  {out_file.name} already exists - reused")
        return
    started = time.time()
    tmp = out_file.with_suffix(".tmp")
    with raw.open(name) as f:
        reader = pacsv.open_csv(
            f,
            read_options=pacsv.ReadOptions(block_size=64 << 20),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(column_types=column_types),
        )
        writer, rows = None, 0
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
            print(f"  {name}: {rows:,} rows", end="\r")
        writer.close()
    tmp.replace(out_file)
    print(f"  {name}: {rows:,} rows -> {out_file.name} ({time.time() - started:.0f}s)")


def build(source, out_dir, cutoff, history_weeks):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = RawData(source)
    started = time.time()

    print("1/4 converting the raw CSVs to parquet")
    string = pa.string()
    csv_to_parquet(raw, "transactions_train.csv", out / "transactions.parquet",
                   {"t_dat": pa.date32(), "customer_id": string, "article_id": string,
                    "price": pa.float64(), "sales_channel_id": pa.int32()})
    csv_to_parquet(raw, "articles.csv", out / "articles.parquet", {"article_id": string})
    csv_to_parquet(raw, "customers.csv", out / "customers.parquet", {"customer_id": string})

    def p(name):
        return str(out / name).replace("\\", "/")

    db = duckdb.connect()
    db.execute("SET memory_limit='3GB'")
    db.execute(f"SET temp_directory='{p('.duckdb_tmp')}'")
    db.execute("SET preserve_insertion_order=false")

    print("2/4 cleaning + missing values + time split")
    median_age = db.execute(f"SELECT CAST(median(age) AS INT) FROM '{p('customers.parquet')}'").fetchone()[0]
    db.execute(f"""
        CREATE TABLE customers AS
        SELECT customer_id,
               CAST(age IS NULL AS INT)                                  AS age_missing,
               COALESCE(age, {median_age})                               AS age,
               COALESCE(club_member_status, 'UNKNOWN')                   AS club_member_status,
               COALESCE(upper(fashion_news_frequency), 'NONE')           AS fashion_news_frequency
        FROM '{p('customers.parquet')}'""")
    db.execute(f"""
        CREATE VIEW train AS
        SELECT *, month(t_dat) AS month,
               dayofweek(t_dat) + 1 AS day_of_week,              -- Sunday = 1, same as Spark
               date_diff('day', t_dat, DATE '{cutoff}') AS days_ago
        FROM '{p('transactions.parquet')}' WHERE t_dat < DATE '{cutoff}'""")
    db.execute(f"CREATE VIEW test AS SELECT * FROM '{p('transactions.parquet')}' WHERE t_dat >= DATE '{cutoff}'")

    print("3/4 contextual features (recency, popularity over time) + vocabularies")
    db.execute("""
        CREATE TABLE customer_feats AS
        SELECT customer_id, count(*) AS n_purchases, count(DISTINCT article_id) AS n_unique_items,
               avg(price) AS avg_price, min(days_ago) AS days_since_last_purchase
        FROM train GROUP BY customer_id""")
    db.execute("""
        CREATE TABLE item_popularity AS
        SELECT article_id,
               CAST(sum(CASE WHEN days_ago <= 7  THEN 1 ELSE 0 END) AS BIGINT) AS purchases_last_7d,
               CAST(sum(CASE WHEN days_ago <= 28 THEN 1 ELSE 0 END) AS BIGINT) AS purchases_last_28d,
               count(*) AS purchases_all_time
        FROM train GROUP BY article_id""")
    db.execute("CREATE TABLE user_vocab AS SELECT customer_id, CAST(row_number() OVER (ORDER BY customer_id) AS INT) AS user_idx FROM (SELECT DISTINCT customer_id FROM customers)")
    db.execute(f"CREATE TABLE item_vocab AS SELECT article_id, CAST(row_number() OVER (ORDER BY article_id) AS INT) AS item_idx FROM (SELECT DISTINCT article_id FROM '{p('articles.parquet')}')")

    db.execute("""
        CREATE TABLE user_table AS
        SELECT v.customer_id, v.user_idx, c.age, c.club_member_status, c.fashion_news_frequency, c.age_missing,
               COALESCE(f.n_purchases, 0) AS n_purchases, COALESCE(f.n_unique_items, 0) AS n_unique_items,
               COALESCE(f.avg_price, 0.0) AS avg_price,
               COALESCE(f.days_since_last_purchase, 999) AS days_since_last_purchase,
               CAST(f.n_purchases IS NULL AS INT) AS is_new_user
        FROM user_vocab v LEFT JOIN customers c USING (customer_id) LEFT JOIN customer_feats f USING (customer_id)""")
    cats = ", ".join(f"COALESCE(a.{c}, 'UNKNOWN') AS {c}" for c in ITEM_CAT_COLS)
    db.execute(f"""
        CREATE TABLE item_table AS
        SELECT v.article_id, v.item_idx, {cats},
               COALESCE(pop.purchases_last_7d, 0) AS purchases_last_7d,
               COALESCE(pop.purchases_last_28d, 0) AS purchases_last_28d,
               COALESCE(pop.purchases_all_time, 0) AS purchases_all_time,
               CAST(pop.purchases_all_time IS NULL AS INT) AS is_new_item
        FROM item_vocab v LEFT JOIN '{p('articles.parquet')}' a USING (article_id)
        LEFT JOIN item_popularity pop USING (article_id)""")

    print("4/4 writing the Week 1 output files")
    # Only the recent weeks are needed for training (the model uses TRAIN_WEEKS <= history_weeks);
    # keeping all 31M rows would only cost disk space on a laptop.
    exports = {
        "interactions_train": f"""
            SELECT COALESCE(u.user_idx, 0) AS user_idx, COALESCE(i.item_idx, 0) AS item_idx, t.t_dat,
                   CAST(t.month AS INT) AS month, CAST(t.day_of_week AS INT) AS day_of_week,
                   CAST(t.days_ago AS INT) AS days_ago, t.price, t.sales_channel_id
            FROM train t LEFT JOIN user_vocab u USING (customer_id) LEFT JOIN item_vocab i USING (article_id)
            WHERE t.days_ago <= {history_weeks * 7}""",
        "interactions_test": """
            SELECT COALESCE(u.user_idx, 0) AS user_idx, COALESCE(i.item_idx, 0) AS item_idx, t.t_dat,
                   t.price, t.sales_channel_id
            FROM test t LEFT JOIN user_vocab u USING (customer_id) LEFT JOIN item_vocab i USING (article_id)""",
        "user_table": "SELECT * FROM user_table",
        "item_table": "SELECT * FROM item_table",
        "user_vocab": "SELECT * FROM user_vocab",
        "item_vocab": "SELECT * FROM item_vocab",
        "article_details": f"SELECT article_id, {', '.join(DETAIL_COLS)} FROM '{p('articles.parquet')}'",
        # last 12 distinct articles each customer bought before the cutoff (newest first)
        "user_history": """
            SELECT customer_id, string_agg(article_id, ',' ORDER BY last_bought DESC) AS recent_articles
            FROM (
                SELECT customer_id, article_id, max(t_dat) AS last_bought,
                       row_number() OVER (PARTITION BY customer_id ORDER BY max(t_dat) DESC, article_id) AS rn
                FROM train WHERE days_ago <= 365 GROUP BY customer_id, article_id
            ) WHERE rn <= 12 GROUP BY customer_id""",
    }
    for name, query in exports.items():
        t0 = time.time()
        db.execute(f"COPY ({query}) TO '{p(name + '.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        rows = db.execute(f"SELECT count(*) FROM '{p(name + '.parquet')}'").fetchone()[0]
        print(f"  saved {name}.parquet ({rows:,} rows, {time.time() - t0:.0f}s)")

    vocabularies = {c: [r[0] for r in db.execute(f"SELECT DISTINCT {c} FROM item_table ORDER BY 1").fetchall()]
                    for c in ITEM_CAT_COLS}
    for c in ["club_member_status", "fashion_news_frequency"]:
        vocabularies[c] = [r[0] for r in db.execute(f"SELECT DISTINCT {c} FROM user_table ORDER BY 1").fetchall()]
    (out / "vocabularies.json").write_text(json.dumps(vocabularies, indent=2))
    top12 = [r[0] for r in db.execute(
        "SELECT article_id FROM item_popularity ORDER BY purchases_last_28d DESC, article_id LIMIT 12").fetchall()]
    (out / "fallback_top12.json").write_text(json.dumps(top12))

    stats = db.execute("""SELECT (SELECT count(*) FROM user_table), (SELECT sum(1 - is_new_user) FROM user_table),
                                 (SELECT count(*) FROM item_table), (SELECT count(*) FROM train),
                                 (SELECT count(*) FROM test)""").fetchone()
    build_info = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "builder": "scripts/build_local_data.py (DuckDB)",
        "raw_source": str(Path(source).resolve()),
        "images_source": str(Path(source).resolve()) if raw.has_images() else None,
        "cutoff": cutoff,
        "history_weeks_kept": history_weeks,
        "customers": stats[0], "customers_with_purchases": int(stats[1]), "articles": stats[2],
        "train_transactions": stats[3], "test_transactions": stats[4],
    }
    (out / "build_info.json").write_text(json.dumps(build_info, indent=2))
    db.close()
    for f in out.glob(".duckdb_tmp*"):
        if f.is_dir():
            for g in f.iterdir():
                g.unlink()
            f.rmdir()
    print(json.dumps(build_info, indent=2))
    print(f"done in {(time.time() - started) / 60:.1f} min -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=os.environ.get("HM_RAW_DATA"),
                        help="Kaggle zip file or folder with the 3 CSVs (env HM_RAW_DATA)")
    parser.add_argument("--out", default=config.DATA_PATH, help="output folder (env HM_DATA_PATH)")
    parser.add_argument("--cutoff", default=config.CUTOFF, help="train < cutoff <= test")
    parser.add_argument("--history-weeks", type=int, default=16,
                        help="weeks of training interactions to keep (training uses the last TRAIN_WEEKS)")
    args = parser.parse_args()
    if not args.source:
        parser.error("--source is required (path to the Kaggle zip or to a folder with the CSVs)")
    build(args.source, args.out, args.cutoff, args.history_weeks)


if __name__ == "__main__":
    main()
