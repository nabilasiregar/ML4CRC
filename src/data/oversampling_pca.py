from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib

def apply_pca_to_data(features_path, scaler, pca, output_path):
    X = pd.read_csv(features_path)
    
    # Standardize the features using the scaler
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Generate PCA column names based on the number of components
    pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    pd.DataFrame(X_pca, columns=pca_columns).to_csv(output_path, index=False)

def process_data_with_pca_and_smote(train_features_path, train_labels_path, val_features_path, test_features_path, output_dir, n_components=None):
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path).squeeze()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)
    
    # Generate PCA column names for the training data
    pca_columns = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
    pd.DataFrame(X_train_smote, columns=pca_columns).to_csv(f"{output_dir}/train_features_smote_pca.csv", index=False)
    y_train_smote.to_csv(f"{output_dir}/train_labels_pca.csv", index=False)
    
    # Apply PCA and save for validation and test data
    for data_path, output_name in [(val_features_path, "val_features_pca.csv"), (test_features_path, "test_features_pca.csv")]:
        apply_pca_to_data(data_path, scaler, pca, f"{output_dir}/{output_name}")

    # Save the PCA model and scaler for later use
    joblib.dump(pca, f"{output_dir}/pca_model.joblib")
    joblib.dump(scaler, f"{output_dir}/scaler_model.joblib")

output_dir = '../../data/final/'
train_features_path = f"{output_dir}/train_features.csv"
train_labels_path = f"{output_dir}/train_labels.csv"
val_features_path = f"{output_dir}/val_features.csv"
test_features_path = f"{output_dir}/test_features.csv"

process_data_with_pca_and_smote(train_features_path, train_labels_path, val_features_path, test_features_path, output_dir, n_components=0.95)
