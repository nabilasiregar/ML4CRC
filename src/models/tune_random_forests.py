import pandas as pd
from random_forests_model import RandomForestModel

if __name__ == "__main__":
    train_data = pd.read_csv('../../data/processed/train_data.csv')
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1].astype(str)

    val_data = pd.read_csv('../../data/processed/val_data.csv')
    X_val = val_data.iloc[:, :-1]
    y_val = val_data.iloc[:, -1].astype(str)

    model = RandomForestModel(random_state=42)

    # Tune hyperparameters using the validation set for evaluation
    best_params, best_score = model.tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100)

    model.save_model("best_random_forest_model.joblib")

    print(f"Best Model Parameters: {best_params}")
    print(f"Best number of trees: {best_params['n_estimators']}")
    print(f"Best Validation Score: {best_score}")
