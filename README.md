# Board Game Recommender System
**CMPE 256 — Recommender Systems | Team 8**

Dataset: [Board Game Database from BoardGameGeek](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek/data)

---

## Team

| Member | Contribution |
|---|---|
| Yun Ei Hlaing | EDA, Data Cleaning, Item-based CF, DeepFM variant |
| Nhi Nguyen | SVD baseline, BPR-MF baseline (replaced), LightGCN variant |
| Uday Arora | Preprocessing pipeline, Popularity baseline, LLM-hybrid variant |

---

## Dataset

22k board games, 272k users, 18.7M interactions after filtering. Rich side information including mechanics, themes, subcategories, designers and publishers per game.

---

## Getting Started

**1. DataPreprocessing.ipynb**
Download the raw dataset from Kaggle and place CSVs under `dataset/`. Run this notebook top to bottom, it produces all processed files under `dataset/processed/`. At the end of the notebook the processed data is uploaded to a shared Google Drive as a zip for teammates to use directly without re-running preprocessing. Processed dataset (~500MB): [Google Drive](https://drive.google.com/file/d/1a7kCwqP2Vhlv43eep0ODdzGVR6v2ouP6/view?usp=sharing)

**2. Any baseline or variant notebook**
Each notebook is self-contained and can be run independently. At the top of every notebook the processed dataset is downloaded automatically from the shared Drive link into `dataset/processed/`. No other setup is needed.

---

## Shared Pipeline

`DataPreprocessing.ipynb` produces the following files all variants and baselines load from:

| File | Description |
|---|---|
| `train.csv` | Training interactions, all but last 2 per user |
| `val.csv` | One interaction per user, second to last |
| `test.csv` | One interaction per user, last interaction |
| `user_ratings_cleaned.csv` | Full filtered interaction table |
| `games_cleaned.csv` | Game metadata with irrelevant columns dropped |
| `id_maps.pkl` | user2idx, item2idx, idx2user, idx2item dictionaries |

Preprocessing steps applied: dropped null usernames, dropped high-missing columns, added normalized ratings, filtered users with fewer than 5 ratings and games with fewer than 10, remapped IDs to 0-indexed integers, applied leave-one-out split by row order.

---

## Baselines

| Baseline | Method | HR@10  | NDCG@10 | MRR@10 |
|---|---|--------|---|--------|
| Popularity | Bayesian average ranking | 0.1831 | 0.0946 | 0.0678 |
| SVD | Matrix Factorization | 0.1920 | 0.1034 | 0.0766 |
| Item-based CF | Cosine Similarity | 0.2446 | 0.1749 | 0.1524 | 

---

## Variants

| Variant | Approach |
|---|---|
| LightGCN | Homogeneous GNN over user-item interaction graph |
| DeepFM | Factorization Machine with deep MLP | 
| LLM-hybrid | Sentence transformer embeddings combined with collaborative filtering scores |

---

## Evaluation

All models evaluated using the shared `evaluate()` function defined in `reference_setup_notebook.ipynb`. Protocol: leave-one-out, 99 sampled negatives per test user, metrics reported at K=10.

| Metric | Description |
|---|---|
| HR@10 | Whether test item appears in top 10 |
| NDCG@10 | Rewards ranking the test item higher within top 10 |
| MRR@10 | Reciprocal rank of the test item |
