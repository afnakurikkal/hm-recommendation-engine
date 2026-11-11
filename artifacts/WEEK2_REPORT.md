# Week 2 — Deep Learning Model Architecture

## Implemented
- Context-aware Query Tower and metadata-aware Candidate Tower
- Two-Tower user/item embedding model
- In-batch negative sampling with multi-positive contrastive loss
- Model training and validation
- Full-catalogue Recall@K and NDCG@K evaluation

## Dataset
- Training interactions: 22,349
- Validation interactions: 2,484
- Catalogue articles: 105,542
- Interactions sampled to 25,000 rows for this Colab experiment.

## Evaluation
- Recall@5: 0.0000
- Recall@10: 0.0004
- NDCG@5: 0.0000
- NDCG@10: 0.0001

## Scope note
Metrics describe the sampled experiment, not a full-transaction production run.
Raw customer data and source CSV files are not included in the artifacts.