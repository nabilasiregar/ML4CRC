import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

class RandomForestModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train, n_estimators=100, max_features='auto', evaluate_on_train=False):
        """
        Train the model and optionally evaluate it on the training set.
        
        :param X_train: Training features.
        :param y_train: Training target variable.
        :param n_estimators: Number of trees in the forest.
        :param max_features: The number of features to consider when looking for the best split.
        :param evaluate_on_train: If True, evaluate the model on the training data.
        :return: Training F1 score if evaluate_on_train is True, otherwise None.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        if evaluate_on_train:
            y_pred_train = self.model.predict(X_train)
            train_f1 = f1_score(y_train, y_pred_train, average='macro')
            return train_f1

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return f1_score(y_test, predictions, average='macro')

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def tune_hyperparameters(self, X_train, y_train, X_test, y_test, n_trials=100):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_features = trial.suggest_categorical('max_features', ['sqrt', None])
            self.train(X_train, y_train, n_estimators=n_estimators, max_features=max_features)
            return self.evaluate(X_test, y_test)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_trial.params
        best_score = study.best_trial.value
        
        # Retrain the model with the best parameters found and evaluate on training set
        train_f1 = self.train(X_train, y_train, n_estimators=best_params['n_estimators'], max_features=best_params['max_features'], evaluate_on_train=True)

        return best_params, best_score, train_f1
