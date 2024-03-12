import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, n_estimators_range, random_state=42):
        self.n_estimators_range = n_estimators_range
        self.random_state = random_state
        self.best_model = None
        self.accuracies = []

    def train(self, X_train, y_train):
        for n_estimators in self.n_estimators_range:
            rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state)
            rf_clf.fit(X_train, y_train)
            self.accuracies.append(rf_clf.score(X_train, y_train))

            if self.best_model is None or rf_clf.score(X_train, y_train) > self.best_model['accuracy']:
                self.best_model = {'model': rf_clf, 'n_estimators': n_estimators, 'accuracy': rf_clf.score(X_train, y_train)}
        
        print(f"Best Model: {self.best_model['n_estimators']} Trees, Accuracy: {self.best_model['accuracy']}")

    def save_model(self, filepath):
        joblib.dump(self.best_model['model'], filepath)


if __name__ == "__main__":
    train_data = pd.read_csv('../../data/processed/train_data.csv')
    train_data = train_data.dropna(subset=['msi_status'])

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1].astype(str)

    model_trainer = RandomForestModel(n_estimators_range=range(1, 101))
    model_trainer.train(X_train, y_train)
    model_trainer.save_model('random_forest_model.joblib')
