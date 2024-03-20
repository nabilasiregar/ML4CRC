import pandas as pd
from random_forests_model import RandomForestModel
import pdb

def load_data(features_path, labels_path):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()
    return X, y

if __name__ == "__main__":
    train_features_path = '../../data/final/train_features_smote.csv'
    train_labels_path = '../../data/final/train_labels_smote.csv'
    val_features_path = '../../data/final/val_features.csv'
    val_labels_path = '../../data/final/val_labels.csv'
    
    X_train, y_train = load_data(train_features_path, train_labels_path)
    X_val, y_val = load_data(val_features_path, val_labels_path)
    
    model = RandomForestModel(random_state=42)
    best_params, best_val_score, train_f1 = model.tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100)
    
    model_filename = "best_random_forest_model_without_pca.joblib"
    model.save_model(model_filename)
    
    print(f"Training F1 Score (Macro Average): {train_f1}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Validation F1 Score: {best_val_score}")
