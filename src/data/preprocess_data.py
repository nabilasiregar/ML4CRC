import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pdb

def preprocess_data(rna_data_path, prediction_data_path, output_dir):
    rna_data = pd.read_csv(rna_data_path, header=0, index_col=0).transpose()
    prediction_data = pd.read_csv(prediction_data_path, header=None)
    prediction_data.columns = ['SampleID'] + prediction_data.iloc[0, 1:].tolist()
    prediction_data = prediction_data[1:]

    merged_data = pd.merge(prediction_data, rna_data, left_on='SampleID', right_index=True)

    # Handling missing values
    filtered_data = merged_data[(merged_data['msi_status'] != 'Indeterminate') & (merged_data['msi_status'].notna())]
   
    features = filtered_data.iloc[:, 11:]  # features
    labels = filtered_data.iloc[:, 2]  # msi_status column

    # Split into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    gene_names = list(rna_data.columns)

    pd.DataFrame(X_train_scaled, columns=gene_names).to_csv(f'{output_dir}/train_features.csv', index=False)
    y_train.to_csv(f'{output_dir}/train_labels.csv', index=False)
    pd.DataFrame(X_val_scaled, columns=gene_names).to_csv(f'{output_dir}/val_features.csv', index=False)
    y_val.to_csv(f'{output_dir}/val_labels.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=gene_names).to_csv(f'{output_dir}/test_features.csv', index=False)
    y_test.to_csv(f'{output_dir}/test_labels.csv', index=False)

rna_data_path = '../../data/raw/tcga_rna_count_data_crc.csv'
prediction_data_path = '../../data/raw/prediction_file_crc.csv'
output_dir = '../../data/final/'

preprocess_data(rna_data_path, prediction_data_path, output_dir)
