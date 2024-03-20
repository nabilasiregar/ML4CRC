import pandas as pd
import joblib
import hydra
from hydra import utils
from omegaconf import DictConfig
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import optuna

class SVMModel:
    def __init__(self, C=1.0, gamma='scale', kernel='rbf', random_state=None):
        self.model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return f1_score(y_test, predictions, average='macro')

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

def load_data(features_path, labels_path):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()
    return X, y

@hydra.main(config_path="../../conf/", config_name="config")
def tune_svm(cfg: DictConfig):
    def objective(trial):
        C = trial.suggest_loguniform('svm.C', 1e-3, 1e3)
        gamma = trial.suggest_categorical('svm.gamma', ['scale', 'auto'])
        kernel = trial.suggest_categorical('svm.kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
        
        model = SVMModel(C=C, gamma=gamma, kernel=kernel, random_state=42)
        
        train_features_path = utils.to_absolute_path(cfg.data.train_features)
        train_labels_path = utils.to_absolute_path(cfg.data.train_labels)
        val_features_path = utils.to_absolute_path(cfg.data.val_features)
        val_labels_path = utils.to_absolute_path(cfg.data.val_labels)

        X_train, y_train = load_data(train_features_path, train_labels_path)
        X_val, y_val = load_data(val_features_path, val_labels_path)
        
        model.train(X_train, y_train)
        return model.evaluate(X_val, y_val)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_params = {k.replace("svm.", ""): v for k, v in study.best_trial.params.items()}
    best_model = SVMModel(**best_params, random_state=42)
    
    X_train, y_train = load_data(utils.to_absolute_path(cfg.data.train_features), utils.to_absolute_path(cfg.data.train_labels))
    
    best_model.train(X_train, y_train)
    best_model.save_model('best_svm_model.joblib')

    print("Best parameters:", best_params)
    print("Best F1 score:", study.best_value)

if __name__ == "__main__":
    tune_svm()
