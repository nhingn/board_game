# Board Game Recommender System
**CMPE 256 — Recommender Systems | Team 8**

Dataset: [Board Game Database from BoardGameGeek](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek/data)

---

## Team

| Member | Contribution                                                              |
|---|---------------------------------------------------------------------------|
| Yun Ei Hlaing | EDA, data cleaning, Item-based CF baseline, DeepFM variant             |
| Nhi Nguyen | SVD baseline, LightGCN variant                                            |
| Uday Arora | Preprocessing pipeline, Popularity baseline, LLM-hybrid variant |

## Dataset

22k board games, 272k users, 18.7M interactions after filtering. Rich side information including mechanics, themes, subcategories, designers and publishers per game.

---

## Getting Started

**1. DataPreprocessing.ipynb**
Download the raw dataset from Kaggle and place CSVs under `dataset/`. Run this notebook top to bottom — it produces all processed files under `dataset/processed/`. At the end of the notebook the processed data is uploaded to a shared Google Drive as a zip for teammates to use directly without re-running preprocessing. Processed dataset (~500MB): [Google Drive](https://drive.google.com/file/d/1a7kCwqP2Vhlv43eep0ODdzGVR6v2ouP6/view?usp=sharing)

**2. Any baseline or variant notebook**
Each notebook is self-contained and can be run independently. At the top of every notebook the processed dataset is downloaded automatically from the shared Drive link into `dataset/processed/`. For a template showing how to load data and use the shared evaluation function, refer to `reference_setup_notebook.ipynb`.

**3. LLM-Hybrid variant only**
Requires SVD latent factor matrices in addition to the processed dataset. SVD matrices (~50MB): [Google Drive](https://drive.google.com/file/d/1N_U-IF-HAovnIWO62iPcIKzBsYNjguxd/view?usp=sharing) — downloaded automatically at the top of the LLM-Hybrid notebook.

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
| `splits.pkl` | Train/val/test dataframes in a single pickle file |

Preprocessing steps applied: dropped null usernames, dropped high-missing columns, added normalized ratings, filtered users with fewer than 5 ratings and games with fewer than 10, remapped IDs to 0-indexed integers, applied leave-one-out split by row order.

---

## Models

| Model | Type | Approach |
|---|---|---|
| Popularity | Baseline | Bayesian average ranking |
| SVD | Baseline | Matrix factorization with mean-centered ratings |
| Item-based CF | Baseline | Cosine similarity over user-item interaction matrix |
| LightGCN | Variant — Nhi Nguyen | Homogeneous GNN over user-item interaction graph |
| DeepFM | Variant — Yun Ei Hlaing | Factorization machine with deep MLP |
| LLM-Hybrid | Variant — Uday Arora | Sentence transformer embeddings combined with SVD collaborative filtering scores |

---

## Results

| Model | HR@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| Popularity | 0.1831 | 0.0946 | 0.0680 |
| SVD | 0.2021 | 0.1089 | 0.0808 |
| Item-based CF | 0.2445 | 0.1748 | 0.1522 |
| LightGCN | 0.2790 | 0.1923 | 0.1671 |
| DeepFM | 0.4257 | 0.2230 | 0.1618 |
| LLM-Hybrid | 0.2401 | 0.1259 | 0.0915 |

---

## Evaluation

All models evaluated using the shared `evaluate()` function defined in `reference_setup_notebook.ipynb`. Protocol: leave-one-out, 99 sampled negatives per test user, metrics reported at K=10. Validation items are excluded from the negative sampling pool to avoid evaluation leakage.

K=10 was chosen as a realistic recommendation list size given users have rated 46 games on average, and is consistent with standard evaluation practice in recommender systems literature.

| Metric | Description |
|---|---|
| HR@10 | Whether test item appears in top 10 |
| NDCG@10 | Rewards ranking the test item higher within top 10 |
| MRR@10 | Reciprocal rank of the test item |
