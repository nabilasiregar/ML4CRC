import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    # Paths to your data
    train_features_path = '../../data/final/train_features_80:20_smote.csv'
    train_labels_path = '../../data/final/train_labels_80:20_smote.csv'
    selected_features_path = '../../data/final/selected_features.csv'

    # Load the training data
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path).squeeze()

    # Check if selected_features.csv exists
    if os.path.exists(selected_features_path):
        print("Selected features file found. Using selected features for training.")
        selected_features = pd.read_csv(selected_features_path, header=None).squeeze()
        X_train = X_train[selected_features.values]
    else:
        print("Selected features file not found. Using all features for training.")

    # Define the model with your best hyperparameters
    best_hyperparameters = {
        "n_estimators": 427,  
        "max_depth": 15,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": None,
        "bootstrap": True 
    }
    model = RandomForestClassifier(random_state=42, **best_hyperparameters)

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, 'rf_model_100features.joblib')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_and_save_model()

