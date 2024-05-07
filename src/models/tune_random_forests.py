import pandas as pd
from random_forests_model import RandomForestModel
import pdb

def load_data(features_path, labels_path):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()
    return X, y

def main():
    train_features_path = '../../data/final/train_features_80:20_smote.csv'
    train_labels_path = '../../data/final/train_labels_80:20_smote.csv'
    
    X_train, y_train = load_data(train_features_path, train_labels_path)
    
    model = RandomForestModel(random_state=42)
    best_params, best_score, train_f1 = model.tune_hyperparameters(X_train, y_train, n_trials=100)
    
    model_filename = "rf_model.joblib"
    model.save_model(model_filename)
    
    print(f"Training F1 Score (Macro Average): {train_f1}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation F1 Score: {best_score}")

if __name__ == "__main__":
    main()
