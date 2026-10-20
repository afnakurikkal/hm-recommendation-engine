# H&M Context-Aware Neural Recommendation Engine

Two-tower deep learning recommender (TensorFlow Recommenders) built on the
H&M Personalized Fashion Recommendations dataset. Uses user metadata,
purchase history and item context to give personalized top-K recommendations.

**Tech stack:** Python, PySpark, TensorFlow / TFRS, Redis, FastAPI, Apache Airflow

## Timeline
| Week | Work | Owner |
|------|------|-------|
| 1 | Data processing & feature engineering (PySpark) | <afna kurikkal> |
| 2 | Two-tower model training & evaluation | <mariyaShaikh> |
| 3 | Model export, Redis feature store, ANN search | |
| 4 | FastAPI service, Airflow pipelines, load testing | |

## Getting the data
1. Get a Kaggle API token (Kaggle → Settings → API → Create New Token).
2. Open `notebooks/week1_data_processing_final.ipynb` in Google Colab.
3. Run all cells. It downloads the data (only if missing) into
   `MyDrive/hm_project/` and creates all processed files.

Processed data is shared in Google Drive: <https://drive.google.com/drive/folders/1g7vdWBCzeLz01PtZKgRqC0-OtCwYVptW?usp=drive_link>

> Never commit `kaggle.json` or data files. They are blocked by `.gitignore`.

## Week 1 output files
All in `MyDrive/hm_project/`. **Index `0` = unknown user/item (cold-start).**
Train = before 2020-09-16, test = last week (2020-09-16 to 2020-09-22).

| File | Columns | Used for |
|------|---------|----------|
| `interactions_train.parquet` | user_idx, item_idx, t_dat, month, day_of_week, days_ago, price, sales_channel_id | Model training |
| `interactions_test.parquet` | user_idx, item_idx, t_dat, price, sales_channel_id | Evaluation (Recall@K, NDCG) |
| `user_table.parquet` | customer_id, user_idx, age, club_member_status, fashion_news_frequency, age_missing, n_purchases, n_unique_items, avg_price, days_since_last_purchase, is_new_user | Query tower features |
| `item_table.parquet` | article_id, item_idx, product_type_name, colour_group_name, department_name, garment_group_name, purchases_last_7d, purchases_last_28d, purchases_all_time, is_new_item | Candidate tower features |
| `user_vocab.parquet` / `item_vocab.parquet` | real ID ↔ index | Mapping IDs to embeddings |
| `vocabularies.json` | category value lists | StringLookup layers |
| `fallback_top12.json` | 12 popular article_ids | Cold-start fallback |

## Repo structure
notebooks/   Colab notebooks for each week