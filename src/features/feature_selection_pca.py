import pandas as pd
import joblib
import sys
sys.path.append('../models')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from random_forests_model import RandomForestModel

def apply_pca_and_oversample(train_features_path, train_labels_path, output_dir, n_components=5):
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path).squeeze()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)

    pd.DataFrame(X_train_smote).to_csv(f"{output_dir}/train_features_smote_top5_pca.csv", index=False)
    y_train_smote.to_csv(f"{output_dir}/train_labels_smote.csv", index=False)

    joblib.dump(pca, f"{output_dir}/pca_top5_model.joblib")
    joblib.dump(scaler, f"{output_dir}/scaler_model.joblib")

def retrain_model_with_selected_features(train_features_path, train_labels_path, output_model_path):
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path).squeeze()

    # Create and train the model
    model = RandomForestModel(random_state=42)
    model.train(X_train, y_train, max_features='sqrt') 

    # Save the trained model
    joblib.dump(model, output_model_path)

if __name__ == "__main__":
    output_dir = '../../data/final/'
    train_features_path = f"{output_dir}/train_features.csv"
    train_labels_path = f"{output_dir}/train_labels.csv"
    model_path = "best_random_forest_top5.joblib"

    # Apply PCA and oversampling
    apply_pca_and_oversample(train_features_path, train_labels_path, output_dir)

    # Retrain the model with the top 5 PCA components
    retrain_model_with_selected_features(f"{output_dir}/train_features_smote_top5_pca.csv", f"{output_dir}/train_labels_smote.csv", model_path)
