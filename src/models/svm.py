import pandas as pd
import numpy as np
import optuna
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from joblib import dump


train_features_path = '../../data/final/train_features_80:20_smote.csv'
train_labels_path = '../../data/final/train_labels_80:20_smote.csv'
test_features_path = '../../data/final/test_features_80:20.csv'
test_labels_path = '../../data/final/test_labels_80:20.csv'

X_train = pd.read_csv(train_features_path)
y_train = pd.read_csv(train_labels_path).squeeze()
X_test = pd.read_csv(test_features_path)
y_test = pd.read_csv(test_labels_path).squeeze()

# List of relevant genes (features) to include
selected_genes = [
    'ATOH1', 'GRM8', 'PCDH8', 'CEL', 'CFTR', 'ENGASE', 'GP2', 'C5orf52', 'SATB2', 'CDX2',
    'CYP2B6', 'NKD1', 'SMAD2', 'MTOR', 'NFE2L2', 'RB1', 'KEAP1', 'TERT', 'RASA1', 'CDC73',
    'CTNNA1', 'ERBB4', 'CD8A', 'PRF1', 'GZMA', 'GZMB', 'CX3CL1', 'CXCL9', 'CXCL10', 'IFNG',
    'IL1B', 'LAG3', 'CTLA4', 'CD274', 'PDCD1', 'TIGIT', 'IDO1', 'PDCD1LG2', 'VEGFA', 'VEGFB',
    'VEGFC', 'VEGFD'
]

# Select only the relevant features
X_train_selected = X_train[selected_genes]
X_test_selected = X_test[selected_genes]

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-3, 1e3)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e-1)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])

    model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = []
    for train_index, test_index in cv.split(X_train_selected, y_train):
        X_train_fold, X_test_fold = X_train_selected.iloc[train_index], X_train_selected.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)
        f1 = f1_score(y_test_fold, y_pred_fold, average='macro')
        f1_scores.append(f1)

    average_f1 = np.mean(f1_scores)
    # Set the F1 score as a user attribute
    trial.set_user_attr('f1_score', average_f1)

    return average_f1

def save_trial_callback(study, trial):
    trial_record = {
        "trial_number": trial.number,
        "C_value": trial.params['C'],
        "gamma_value": trial.params['gamma'],
        "kernel_value": trial.params['kernel'],
        "f1_score": trial.user_attrs['f1_score']
    }
    
    trial_df = pd.DataFrame([trial_record])
    trial_df.to_csv('svm_log.csv', mode='a', index=False, header=not pd.io.common.file_exists('svm_log.csv'))

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)

study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=100, callbacks=[save_trial_callback])

# After optimization, you can directly save the best model as before
best_trial = study.best_trial
print(f"Best trial F1 score: {best_trial.value}")
print(f"Best hyperparameters: {best_trial.params}")

best_model = SVC(**best_trial.params, random_state=42)
best_model.fit(X_train_selected, y_train)
dump(best_model, 'best_svm_model.joblib')