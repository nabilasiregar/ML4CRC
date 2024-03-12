import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForestModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def tune_hyperparameters(self, X_train, y_train, X_test, y_test, n_trials=100):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            self.train(X_train, y_train, n_estimators)
            return self.evaluate(X_test, y_test)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value
        # After tuning, train the model again with the best parameters found
        self.train(X_train, y_train, n_estimators=best_params['n_estimators'])

        return best_params, best_score
