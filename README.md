# Kaggle Student Health — Multi-Class Health Condition Classification

Predicting a student's health condition (`fit`, `at-risk`, `unhealthy`) from lifestyle and physiological features, built for a Kaggle Playground-style tabular competition. This project was built to practice end-to-end tabular ML: data merging, feature engineering, class-imbalance handling, cross-validation, and prediction ensembling.

## Problem

Given student lifestyle data (sleep duration, stress level, physical activity level, and related features), predict one of three health condition classes. The target classes are imbalanced, so the pipeline is built around balanced-accuracy as the evaluation metric rather than raw accuracy.

## Data

- `train.csv` / `test.csv` — the official competition train/test files.
- `student_health_dataset_50k.csv` — an additional **original** public dataset I downloaded from Kaggle (~50k student health records) that overlaps in schema with the competition data. I merged this in as extra training signal to reduce variance on the minority classes.

## Approach

1. **Merge & clean** — Concatenate `train.csv`, `test.csv`, and the original 50k dataset into one frame, tagging each row with its source so the correct rows can be split back out after feature engineering.
2. **Feature engineering** — Threshold-based binary features found to be predictive during EDA, e.g.:
   - `is_sleep_under_6`, `is_sleep_under_7` (sleep deprivation flags)
   - `is_stress_low`, `is_stress_high`
   - `is_activity_active`, `is_activity_sedentary`
   Categorical columns are cast to `category` dtype for native handling by XGBoost.
3. **Class imbalance handling** — `compute_sample_weight('balanced', y)` is used so minority classes (e.g. `unhealthy`) aren't drowned out by the majority class.
4. **Modeling** — `XGBClassifier` (GPU-enabled) as the primary model, with CatBoost also explored (`src/train_catboost.py`).
5. **Validation** — Stratified K-Fold cross-validation, scored on **balanced accuracy**, to get an honest out-of-fold estimate before touching the leaderboard.
6. **Seed blending** — Instead of a single model, 5 XGBoost models are trained across different random seeds (`42, 1337, 2026, 777, 999`), and their predicted class probabilities are averaged before taking the final `argmax`. This reduces prediction variance from any single seed's random initialization/row sampling.
7. **Ensembling** — Predictions from the seed-blended XGBoost run and the CatBoost model are combined (`src/ensemble_pipeline.py`) into a final ensemble submission.
8. **Submission generation** — Final predictions are decoded back from label-encoded integers to the original class names and written out as a Kaggle-format CSV.

## Project Structure

```
├── catboost_info/                     # CatBoost training logs/artifacts
├── data/
│   ├── student_health_dataset_50k.csv # Additional public dataset for training augmentation
│   ├── test.csv                       # Competition test data
│   └── train.csv                      # Competition train data
├── src/
│   ├── ensemble_pipeline.py           # Combines model outputs into a final ensemble
│   └── train_catboost.py              # CatBoost training script
├── submissions/
│   ├── final_ensemble_submission.csv  # XGBoost + CatBoost ensemble submission
│   ├── blended_submission.csv         # Seed + model blending experiment
│   └── seed_blended_submission.csv    # 5-seed probability-averaged XGBoost submission
└── README.md
```

## Tech Stack

Python, Pandas, NumPy, scikit-learn, XGBoost, CatBoost

## Key Takeaways

- Practiced balanced-accuracy-aware modeling for an imbalanced multi-class target.
- Implemented seed-averaged probability blending to reduce variance vs. a single model.
- Combined multiple model families (XGBoost, CatBoost) into an ensemble to improve generalization over any single model.
