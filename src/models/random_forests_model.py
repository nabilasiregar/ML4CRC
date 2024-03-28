import joblib
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import csv


class RandomForestModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train, n_estimators=100, max_features=None, max_depth=None, 
              min_samples_split=4, min_samples_leaf=3, bootstrap=True):
        """
        Train the RandomForest model with given hyperparameters.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_features=max_features, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return f1_score(y_test, predictions, average='macro')

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def tune_hyperparameters(self, X_train, y_train, n_trials=100, n_splits=5, csv_file='tuning_log.csv'):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 10, 100)

            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_features=None,  
                max_depth=max_depth,
                min_samples_split=4, 
                min_samples_leaf=3, 
                bootstrap=True, 
                random_state=self.random_state
            )

            # Perform cross-validation
            f1_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=n_splits), scoring='f1_macro')
            mean_f1_score = f1_scores.mean()

            return mean_f1_score

        def trial_callback(study, trial):
            # Append trial results to CSV file
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([trial.number, trial.params['n_estimators'], trial.params['max_depth'], trial.value])

        # Prepare the CSV file for logging
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial', 'n_estimators', 'max_depth', 'f1_score'])

        # Create and optimize the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

        best_params = study.best_params
        best_score = study.best_value

        # Retrain the model on the full dataset with the best parameters
        self.model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            max_features=None,
            min_samples_split=4,
            min_samples_leaf=3,
            bootstrap=True,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)

        # Optionally, evaluate the retrained model's performance on the training set or a separate test set
        train_f1 = self.evaluate(X_train, y_train)
        print("F1 Score after retraining on the full training set:", train_f1)

        # Save the retrained model
        self.save_model('../../data/jobs/retrained_rf_model.joblib')

        return best_params, best_score, train_f1
