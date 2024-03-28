from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib

def oversample_data(train_features_path, train_labels_path, output_dir):
    # Load the data
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path).squeeze()

    # Capture feature names
    feature_names = X_train.columns.tolist()

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Convert scaled features back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled_df, y_train)
    
    # Save the oversampled data with feature names
    pd.DataFrame(X_train_smote, columns=feature_names).to_csv(f"{output_dir}/train_features_80:20_smote.csv", index=False)
    pd.DataFrame(y_train_smote, columns=[y_train.name]).to_csv(f"{output_dir}/train_labels_80:20_smote.csv", index=False)
    
    # Save the scaler model for later use
    scaler_model_path = f"{output_dir}/scaler_model_80:20.joblib"
    joblib.dump(scaler, scaler_model_path)

    return smote

# Define paths
output_dir = '../../data/final/'
train_features_path = f"{output_dir}/train_features_80:20.csv"
train_labels_path = f"{output_dir}/train_labels_80:20.csv"

# Execute the function
smote = oversample_data(train_features_path, train_labels_path, output_dir)
