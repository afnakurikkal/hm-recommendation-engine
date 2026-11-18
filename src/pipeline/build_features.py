"""Week 1 feature pipeline as a Spark job (same logic as notebooks/week1_data_processing_final.ipynb).

Used by the Airflow DAGs. Reads the raw H&M CSVs (or the cached transactions
parquet), cleans them, splits train/test by time, builds the contextual
features and vocabularies, and writes the parquet/JSON files Week 2-4 use.

Run:  spark-submit src/pipeline/build_features.py     (or: python -m src.pipeline.build_features)
"""
import json
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType, StructField, StructType

DATA = os.environ.get("HM_DATA_PATH", "/content/drive/MyDrive/hm_project")
CUTOFF = os.environ.get("HM_CUTOFF", "2020-09-16")
ITEM_CAT_COLS = ["product_type_name", "colour_group_name", "department_name", "garment_group_name"]


def path(name):
    return os.path.join(DATA, name)


def main():
    spark = (SparkSession.builder.appName("hm_build_features")
             .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "8g")).getOrCreate())

    # ---------------- load (explicit schema, article_id as string keeps leading zeros)
    schema = StructType([
        StructField("t_dat", DateType()), StructField("customer_id", StringType()),
        StructField("article_id", StringType()), StructField("price", DoubleType()),
        StructField("sales_channel_id", IntegerType()),
    ])
    if not os.path.exists(path("transactions.parquet")):
        spark.read.csv(path("transactions_train.csv"), header=True, schema=schema) \
            .write.mode("overwrite").parquet(path("transactions.parquet"))
    transactions = spark.read.parquet(path("transactions.parquet"))
    customers = spark.read.csv(path("customers.csv"), header=True, inferSchema=True)
    articles = spark.read.csv(path("articles.csv"), header=True, multiLine=True, escape='"')

    # ---------------- missing values
    median_age = int(customers.approxQuantile("age", [0.5], 0.01)[0])
    customers = (customers
                 .withColumn("age_missing", F.col("age").isNull().cast("int"))
                 .withColumn("fashion_news_frequency", F.upper(F.col("fashion_news_frequency")))
                 .fillna({"age": median_age, "FN": 0, "Active": 0,
                          "club_member_status": "UNKNOWN", "fashion_news_frequency": "NONE"}))

    # ---------------- time split + contextual features
    train = (transactions.filter(F.col("t_dat") < CUTOFF)
             .withColumn("month", F.month("t_dat"))
             .withColumn("day_of_week", F.dayofweek("t_dat"))
             .withColumn("days_ago", F.datediff(F.lit(CUTOFF).cast("date"), F.col("t_dat"))))
    test = transactions.filter(F.col("t_dat") >= CUTOFF)

    customer_feats = train.groupBy("customer_id").agg(
        F.count("*").alias("n_purchases"),
        F.countDistinct("article_id").alias("n_unique_items"),
        F.avg("price").alias("avg_price"),
        F.min("days_ago").alias("days_since_last_purchase"),
    )
    item_popularity = train.groupBy("article_id").agg(
        F.sum(F.when(F.col("days_ago") <= 7, 1).otherwise(0)).alias("purchases_last_7d"),
        F.sum(F.when(F.col("days_ago") <= 28, 1).otherwise(0)).alias("purchases_last_28d"),
        F.count("*").alias("purchases_all_time"),
    )

    # ---------------- vocabularies (all customers / articles, 0 = unknown)
    user_vocab = customers.select("customer_id").distinct() \
        .withColumn("user_idx", F.row_number().over(Window.orderBy("customer_id")))
    item_vocab = articles.select("article_id").distinct() \
        .withColumn("item_idx", F.row_number().over(Window.orderBy("article_id")))

    user_table = (user_vocab
                  .join(customers.select("customer_id", "age", "club_member_status",
                                         "fashion_news_frequency", "age_missing"), "customer_id", "left")
                  .join(customer_feats, "customer_id", "left")
                  .withColumn("is_new_user", F.col("n_purchases").isNull().cast("int"))
                  .fillna({"n_purchases": 0, "n_unique_items": 0, "avg_price": 0.0,
                           "days_since_last_purchase": 999}))
    item_table = (item_vocab
                  .join(articles.select("article_id", *ITEM_CAT_COLS), "article_id", "left")
                  .join(item_popularity, "article_id", "left")
                  .withColumn("is_new_item", F.col("purchases_all_time").isNull().cast("int"))
                  .fillna({"purchases_last_7d": 0, "purchases_last_28d": 0, "purchases_all_time": 0})
                  .fillna("UNKNOWN", subset=ITEM_CAT_COLS))

    interactions_train = (train.join(user_vocab, "customer_id", "left").join(item_vocab, "article_id", "left")
                          .fillna({"user_idx": 0, "item_idx": 0})
                          .select("user_idx", "item_idx", "t_dat", "month", "day_of_week",
                                  "days_ago", "price", "sales_channel_id"))
    interactions_test = (test.join(user_vocab, "customer_id", "left").join(item_vocab, "article_id", "left")
                         .fillna({"user_idx": 0, "item_idx": 0})
                         .select("user_idx", "item_idx", "t_dat", "price", "sales_channel_id"))

    # ---------------- save
    for name, df in [("interactions_train", interactions_train), ("interactions_test", interactions_test),
                     ("user_table", user_table), ("item_table", item_table),
                     ("user_vocab", user_vocab), ("item_vocab", item_vocab)]:
        df.write.mode("overwrite").parquet(path(f"{name}.parquet"))
        print("saved", name)

    vocabularies = {c: sorted(r[0] for r in item_table.select(c).distinct().collect()) for c in ITEM_CAT_COLS}
    for c in ["club_member_status", "fashion_news_frequency"]:
        vocabularies[c] = sorted(r[0] for r in user_table.select(c).distinct().collect())
    with open(path("vocabularies.json"), "w") as f:
        json.dump(vocabularies, f, indent=2)
    top12 = [r["article_id"] for r in item_popularity.orderBy(F.desc("purchases_last_28d")).limit(12).collect()]
    with open(path("fallback_top12.json"), "w") as f:
        json.dump(top12, f)
    print("saved vocabularies.json and fallback_top12.json")
    spark.stop()


if __name__ == "__main__":
    main()
