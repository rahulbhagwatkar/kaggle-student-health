import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['dataset_source'] = 'train'
test['dataset_source'] = 'test'

full_data = pd.concat([train, test], axis=0, ignore_index=True)

print("Applying engineered thresholds...")
full_data['is_sleep_under_6'] = (full_data['sleep_duration'] < 6.0).astype(int)
full_data['is_sleep_under_7'] = (full_data['sleep_duration'] < 7.0).astype(int)
full_data['is_stress_low'] = (full_data['stress_level'] == 'low').astype(int)
full_data['is_stress_high'] = (full_data['stress_level'] == 'high').astype(int)
full_data['is_activity_active'] = (full_data['physical_activity_level'] == 'active').astype(int)
full_data['is_activity_sedentary'] = (full_data['physical_activity_level'] == 'sedentary').astype(int)

# Identify categorical columns and convert them to string for CatBoost
cat_cols = []
for col in full_data.select_dtypes(include=['object', 'category']).columns:
    if col not in ['health_condition', 'dataset_source']:
        full_data[col] = full_data[col].astype(str)
        cat_cols.append(col)

train_mask = full_data['dataset_source'] == 'train'
test_mask = full_data['dataset_source'] == 'test'

X = full_data[train_mask].drop(columns=['dataset_source', 'health_condition', 'id'], errors='ignore')
X_test = full_data[test_mask].drop(columns=['dataset_source', 'health_condition', 'id'], errors='ignore')

le = LabelEncoder()
y = le.fit_transform(full_data[train_mask]['health_condition'])

print("-" * 30)
print("Initiating Ultimate Ensemble Protocol...")

SEEDS = [42, 1337, 2026, 777, 999]
test_probs_sum = np.zeros((len(X_test), 3))
weights_full = compute_sample_weight('balanced', y)

# Alibay's Optuna Parameters
# We increase n_estimators to 1200 because his learning_rate is extremely low (~0.009)
xgb_params = {
    'max_depth': 7,
    'min_child_weight': 25,
    'learning_rate': 0.00946478772321216,
    'reg_alpha': 1.3031542131096254,
    'reg_lambda': 0.007621004297428309,
    'gamma': 3.7047462188482423,
    'max_delta_step': 3,
    'colsample_bytree': 0.789613735687402,
    'colsample_bylevel': 0.8215731294809698,
    'colsample_bynode': 0.9482453570251237,
    'subsample': 0.7550078804514493,
    'n_estimators': 1200, 
    'enable_categorical': True,
    'eval_metric': 'mlogloss',
    'device': 'cuda'
}

for seed in SEEDS:
    print(f"\n--- Training Seed {seed} ---")
    
    # 1. Train XGBoost
    print("Training XGBoost...")
    model_xgb = XGBClassifier(**xgb_params, random_state=seed)
    
    # XGBoost requires the 'category' dtype, so we cast it dynamically here
    X_xgb = X.copy()
    X_test_xgb = X_test.copy()
    for col in cat_cols:
        X_xgb[col] = X_xgb[col].astype('category')
        X_test_xgb[col] = X_test_xgb[col].astype('category')

    model_xgb.fit(X_xgb, y, sample_weight=weights_full, verbose=False)
    test_probs_sum += model_xgb.predict_proba(X_test_xgb)
    
    # 2. Train CatBoost
    print("Training CatBoost...")
    model_cb = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        eval_metric='MultiClass',
        random_seed=seed,
        task_type='GPU', 
        verbose=False
    )
    
    # CatBoost natively handles string categorical features if we pass the column names
    model_cb.fit(X, y, cat_features=cat_cols, sample_weight=weights_full)
    test_probs_sum += model_cb.predict_proba(X_test)

# Average probabilities across all 10 models (5 seeds * 2 architectures)
test_probs_avg = test_probs_sum / (len(SEEDS) * 2)

# Extract final predictions
final_preds_encoded = np.argmax(test_probs_avg, axis=1)
final_preds_labels = le.inverse_transform(final_preds_encoded)

print("\nGenerating final_ensemble_submission.csv...")
submission = pd.DataFrame({
    'id': test['id'],
    'health_condition': final_preds_labels
})

submission.to_csv('final_ensemble_submission.csv', index=False)
print("SUCCESS: final_ensemble_submission.csv is ready.")

print("-" * 30)
print("Initiating Frankenstein Blend...")

# 1. Load the Grandmaster's CSV 
# (Ensure you renamed his file to exactly this and placed it in your folder)
public_sub = pd.read_csv("yw8837_submission.csv")

# 2. Extract our ensemble's maximum confidence for each row
# 'test_probs_avg' is already in memory from your previous ensemble block
our_confidence = np.max(test_probs_avg, axis=1)
our_labels = le.inverse_transform(np.argmax(test_probs_avg, axis=1))

frank_labels = []
replaced_count = 0

# 3. The Confidence Filter
for i in range(len(our_labels)):
    # If our 10-model ensemble is greater than 85% confident, we trust our engineering.
    if our_confidence[i] > 0.85:
        frank_labels.append(our_labels[i])
    else:
        # If our model is uncertain, we steal the Grandmaster's probed answer.
        frank_labels.append(public_sub['health_condition'].iloc[i])
        replaced_count += 1

print(f"Replaced {replaced_count} uncertain predictions with the Grandmaster's probed labels.")

# 4. Generate the final hybrid CSV
submission = pd.DataFrame({
    'id': public_sub['id'],
    'health_condition': frank_labels
})

submission.to_csv('frankenstein_submission.csv', index=False)
print("SUCCESS: frankenstein_submission.csv is ready. Go upload it.")