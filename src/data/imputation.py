import pandas as pd
from sklearn.impute import SimpleImputer

class DataImputer:
    def __init__(self, filepath, output_path):
        self.filepath = filepath
        self.output_path = output_path

    def impute_missing_values(self):
        raw_data = pd.read_csv(self.filepath)
        imputer = SimpleImputer(strategy='most_frequent')
        raw_data['msi_status'] = imputer.fit_transform(raw_data[['msi_status']])
        raw_data.to_csv(self.output_path, index=False)


