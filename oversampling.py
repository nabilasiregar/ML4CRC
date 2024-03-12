import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder

file_path = r'C:...\train_data.csv'
data = pd.read_csv(file_path)

target = "msi_status"

features = data.drop(target, axis=1)
target_variable = data[target]

label_encoder = LabelEncoder()
target_variable_encoded = label_encoder.fit_transform(target_variable)

last_column_index = len(features.columns) - 1
oversampler = SMOTENC(categorical_features=[last_column_index], random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(features, target_variable_encoded)

# # Decode target variable back to original labels if needed
# # y_resampled_original = label_encoder.inverse_transform(y_resampled)


output_feature_file = "resampled_features.csv"
pd.DataFrame(X_resampled).to_csv(output_feature_file, index=False)


output_target_file = "resampled_target.csv"
pd.DataFrame(y_resampled, columns=["msi_status"]).to_csv(output_target_file, index=False)

# Combined feature and target
combined_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=["msi_status"])], axis=1)
output_combined_file = "resampled_data.csv"
combined_data.to_csv(output_combined_file, index=False)
# "MSS" : 0
# "MSI-H" : 1
# "MSI-L" : 2
# nan : 3