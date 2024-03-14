import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(rna_data_path, prediction_data_path, train_data_path, val_data_path, test_data_path, nan_strategy='drop'):
    rna_data = pd.read_csv(rna_data_path)
    prediction_data = pd.read_csv(prediction_data_path)
    
    rna_data.rename(columns={rna_data.columns[0]: 'SampleID'}, inplace=True)
    prediction_data.rename(columns={prediction_data.columns[0]: 'SampleID'}, inplace=True)
    
    transposed_rna_data = rna_data.set_index('SampleID').transpose().reset_index().rename(columns={'index': 'SampleID'})
    merged_data = pd.merge(prediction_data, transposed_rna_data, on='SampleID', how='inner')
    
    if nan_strategy == 'drop':
        filtered_data = merged_data[(merged_data['msi_status'] != 'Indeterminate') & (merged_data['msi_status'].notna())]
    elif nan_strategy == 'impute':
        filtered_data = merged_data[merged_data['msi_status'] != 'Indeterminate']
        imputer = SimpleImputer(strategy='most_frequent')
        filtered_data['msi_status'] = imputer.fit_transform(filtered_data[['msi_status']])
    else:
        raise ValueError(f"Unknown NaN strategy: {nan_strategy}")
    
    features = filtered_data.drop(columns=['SampleID', 'msi_status']).select_dtypes(include=['int64', 'float64'])
    target = filtered_data['msi_status']
    
    # Split the data into training+validation and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Split the training+validation set into individual training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # Standardize the data based on the training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    pd.concat([pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train.reset_index(drop=True)], axis=1).to_csv(train_data_path, index=False)
    pd.concat([pd.DataFrame(X_val_scaled, columns=X_val.columns), y_val.reset_index(drop=True)], axis=1).to_csv(val_data_path, index=False)
    pd.concat([pd.DataFrame(X_test_scaled, columns=X_test.columns), y_test.reset_index(drop=True)], axis=1).to_csv(test_data_path, index=False)

rna_data_path = '../../data/raw/tcga_rna_count_data_crc.csv'
prediction_data_path = '../../data/raw/prediction_file_crc.csv'
train_data_path = '../../data/processed/train_data.csv'
val_data_path = '../../data/processed/val_data.csv'
test_data_path = '../../data/processed/test_data.csv'

preprocess_data(rna_data_path, prediction_data_path, train_data_path, val_data_path, test_data_path, nan_strategy='drop')
