import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

X = pd.read_csv('../../data/final/reduced_training_features.csv')
y = pd.read_csv('../../data/final/train_labels_80:20_smote.csv').squeeze()

best_params = {
    "n_estimators": 427,  
    "max_depth": 15,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "max_features": None,
    "bootstrap": True 
}

fold_options = [2, 3, 4, 5]
all_feature_importances = pd.DataFrame(index=X.columns)

# Perform cross-validation and collect feature importances
for n_folds in fold_options:
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_feature_importances = np.zeros((n_folds, X.shape[1]))

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf = RandomForestClassifier(**best_params, random_state=42)
        rf.fit(X_train, y_train)
        fold_feature_importances[fold_idx] = rf.feature_importances_
    
    # Store the average feature importances for this fold count
    all_feature_importances[f'{n_folds}_folds'] = fold_feature_importances.mean(axis=0)

# Save to CSV
all_feature_importances.to_csv('feature_importances_across_folds.csv')