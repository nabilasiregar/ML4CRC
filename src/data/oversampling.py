import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import LabelEncoder

def oversample_with_smote(file_path):
    train_data = pd.read_csv(file_path)
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    oversampled_train_data = pd.concat([X_res, y_res], axis=1)
    oversampled_file_path = file_path.replace("train_data.csv", "train_data_oversampled.csv")
    oversampled_train_data.to_csv(oversampled_file_path, index=False)

def oversample_with_smotenc(file_path, target, output_feature_file, output_target_file, output_combined_file):
    data = pd.read_csv(file_path)
    features = data.drop(target, axis=1)
    target_variable = data[target]
    label_encoder = LabelEncoder()
    target_variable_encoded = label_encoder.fit_transform(target_variable)
    last_column_index = len(features.columns) - 1
    oversampler = SMOTENC(categorical_features=[last_column_index], random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(features, target_variable_encoded)
    # Combined feature and target
    combined_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=[target])], axis=1)
    combined_data.to_csv(output_combined_file, index=False)

# Example usage
# oversample_with_smote('../../data/processed/train_data.csv')
# oversample_with_smotenc('../../data/processed/train_data.csv', "msi_status", "resampled_features.csv", "resampled_target.csv", "resampled_data.csv")

