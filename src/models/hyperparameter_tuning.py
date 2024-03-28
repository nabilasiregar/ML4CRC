import pandas as pd
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import csv

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath

    def log_trial(self, trial):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([trial.number] + list(trial.params.values()) + [trial.value])

    def create_log_file(self, header):
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 100, 200),
        "max_depth": trial.suggest_int('max_depth', 5, 50),
        "min_samples_split": trial.suggest_int('min_samples_split', 2, 10),
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 10),
        "max_features": trial.suggest_categorical('max_features', [None]),
        "bootstrap": trial.suggest_categorical('bootstrap', [True, False]),
    }
    model = RandomForestClassifier(random_state=42, **params)
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='f1_macro')
    return scores.mean()

def main():
    train_features_path = '../../data/final/train_features_80:20_smote.csv'
    train_labels_path = '../../data/final/train_labels_80:20_smote.csv'
    selected_features_path = 'selected_features.csv'

    # Load the training data and the selected features
    X_train_full = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path).squeeze()
    selected_features = pd.read_csv(selected_features_path, header=None).squeeze()

    # Subset the training data to only include the selected features
    X_train_selected = X_train_full[selected_features.values]

    # Initialize logging
    logger = CSVLogger('tuning_log.csv')
    header = ['trial', 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'bootstrap', 'f1_score']
    logger.create_log_file(header)

    # Hyperparameter tuning
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_selected, y_train), n_trials=100)

    # Log trial results
    for trial in study.trials:
        logger.log_trial(trial)

    best_params = study.best_params
    print(f"Best Parameters: {best_params}")

    # Retrain the model with the best hyperparameters using cross-validation
    final_model = RandomForestClassifier(random_state=42, **best_params)

    # Train a single final model on all data.
    final_model.fit(X_train_selected, y_train)

    # Save the retrained best model
    joblib.dump(final_model, 'rf_model_final.joblib')
    print("Final model trained with the best hyperparameters is saved as 'rf_model_final.joblib'.")

if __name__ == "__main__":
    main()
