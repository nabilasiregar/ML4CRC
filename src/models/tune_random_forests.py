import pandas as pd
from sklearn.model_selection import train_test_split
from random_forests_model import RandomForestModel

if __name__ == "__main__":
    data = pd.read_csv('../../data/processed/train_data.csv').dropna(subset=['msi_status'])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestModel(random_state=42)
    best_params, best_score = model.tune_hyperparameters(X_train, y_train, X_test, y_test, n_trials=100)

    model.save_model("random_forest_model.joblib")
    print(f"Best Model: {best_params['n_estimators']} Trees, Accuracy: {best_score}")