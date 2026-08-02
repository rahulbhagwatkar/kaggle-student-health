import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# 1. LOAD AND MERGE
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# original = pd.read_csv("student_health_dataset_50k.csv") # Uncomment if you downloaded it

train['dataset_source'] = 'train'
test['dataset_source'] = 'test'
# original['dataset_source'] = 'original'

# Concatenate (Include 'original' in this list if you have it)
full_data = pd.concat([train, test], axis=0, ignore_index=True)

# 2. FEATURE ENGINEERING (The Golden Thresholds)
full_data['is_sleep_under_6'] = (full_data['sleep_duration'] < 6.0).astype(int)
full_data['is_sleep_under_7'] = (full_data['sleep_duration'] < 7.0).astype(int)
full_data['is_stress_low'] = (full_data['stress_level'] == 'low').astype(int)
full_data['is_stress_high'] = (full_data['stress_level'] == 'high').astype(int)
full_data['is_activity_active'] = (full_data['physical_activity_level'] == 'active').astype(int)
full_data['is_activity_sedentary'] = (full_data['physical_activity_level'] == 'sedentary').astype(int)

# Convert remaining object columns to category for XGBoost
for col in full_data.select_dtypes(include=['object']).columns:
    if col not in ['health_condition', 'dataset_source']:
        full_data[col] = full_data[col].astype('category')

# 3. SPLIT DATA BACK
train_mask = full_data['dataset_source'].isin(['train', 'original'])
test_mask = full_data['dataset_source'] == 'test'

# Drop ID if it exists so the model doesn't train on it
X = full_data[train_mask].drop(columns=['dataset_source', 'health_condition', 'id'], errors='ignore')
X_test = full_data[test_mask].drop(columns=['dataset_source', 'health_condition', 'id'], errors='ignore')

# Encode the target labels ('fit', 'at-risk', 'unhealthy') to (0, 1, 2)
le = LabelEncoder()
y = le.fit_transform(full_data[train_mask]['health_condition'])
import numpy as np

print("-" * 30)
print("Initiating Seed Blending Protocol...")

# 1. Define your seeds. 5 is a good balance between variance reduction and speed.
SEEDS = [42, 1337, 2026, 777, 999]

# 2. Create an empty array to store the cumulative probabilities
# Shape will be (number of test rows, 3 classes)
test_probs_sum = np.zeros((len(X_test), 3))

# 3. Compute weights for the entire dataset once
weights_full = compute_sample_weight('balanced', y)

# 4. Train a model for every seed
for seed in SEEDS:
    print(f"Training XGBoost with seed {seed}...")
    
    model = XGBClassifier(
        n_estimators=150, # Dropped from 1000. Without early stopping, 1000 will overfit.
        learning_rate=0.05, 
        max_depth=6, 
        enable_categorical=True,
        eval_metric='mlogloss',
        random_state=seed,
        device='cuda' # Remove if not on GPU
    )
    
    # Train on 100% of the training data
    model.fit(X, y, sample_weight=weights_full, verbose=False)
    
    # Extract probabilities instead of hard labels, and add them to the total
    test_probs_sum += model.predict_proba(X_test)

# 5. Average the probabilities across all 5 models
test_probs_avg = test_probs_sum / len(SEEDS)

# 6. For each row, pick the class index (0, 1, or 2) with the highest average probability
final_preds_encoded = np.argmax(test_probs_avg, axis=1)

# 7. Decode back to text labels ('fit', 'at-risk', 'unhealthy')
final_preds_labels = le.inverse_transform(final_preds_encoded)

print("Generating seed_blended_submission.csv...")
# 8. Format the final Kaggle submission
submission = pd.DataFrame({
    'id': test['id'],
    'health_condition': final_preds_labels
})

submission.to_csv('seed_blended_submission.csv', index=False)
print("SUCCESS: seed_blended_submission.csv is ready. Go upload it.")

# 4. STRATIFIED K-FOLD VALIDATION
N_SPLITS = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X)) # To store Out-Of-Fold predictions
fold_scores = []

print("Starting Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Isolate training and validation sets for this fold
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Compute balanced weights to force the model to respect minority classes
    weights = compute_sample_weight('balanced', y_train_fold)

    # Initialize model (add Optuna tuned params here later)
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,  # Keep it shallow to prevent overfitting noise
        enable_categorical=True,
        eval_metric='mlogloss',
        early_stopping_rounds=50,
        random_state=42
    )

    # Train the model
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        sample_weight=weights,
        verbose=False
    )

    # Predict on the isolated validation set
    val_preds = model.predict(X_val_fold)
    oof_preds[val_idx] = val_preds

    # Calculate fold score
    score = balanced_accuracy_score(y_val_fold, val_preds)
    fold_scores.append(score)
    print(f"Fold {fold + 1} Balanced Accuracy: {score:.5f}")

# 5. FINAL EVALUATION
cv_score = balanced_accuracy_score(y, oof_preds)
print("-" * 30)
print(f"Mean Fold Score: {np.mean(fold_scores):.5f}")
print(f"Overall OOF Balanced Accuracy: {cv_score:.5f}")