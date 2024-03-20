import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

class RandomForestModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train, n_estimators=100, max_features='auto', max_depth=None, 
              min_samples_split=2, min_samples_leaf=1, bootstrap=True):
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

    def tune_hyperparameters(self, X_train, y_train, X_test, y_test, n_trials=100):
        def objective(trial):
            # Define the hyperparameter configuration space
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            max_depth = trial.suggest_int('max_depth', 10, 100, log=True) or None
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            bootstrap = trial.suggest_categorical('bootstrap', [True, False])

            # Train the model with suggested hyperparameters
            self.train(
                X_train, y_train, 
                n_estimators=n_estimators, 
                max_features=max_features,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap
            )
            
            # Evaluate the model
            return self.evaluate(X_test, y_test)

        # Create a study object and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value
        
        # Retrain with best parameters and evaluate on the training set
        self.train(
            X_train, y_train, 
            n_estimators=best_params['n_estimators'], 
            max_features=best_params['max_features'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            bootstrap=best_params['bootstrap']
        )
        
        train_f1 = self.evaluate(X_train, y_train)

        return best_params, best_score, train_f1
