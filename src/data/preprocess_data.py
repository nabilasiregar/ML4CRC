import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, raw_file_path=None, imputation_file_path=None, output_folder='../../data/processed', perform_imputation=False, perform_normalization=True, perform_pca=True, split_data=True):
        self.raw_file_path = raw_file_path
        self.imputation_file_path = imputation_file_path
        self.output_folder = output_folder
        self.should_impute = perform_imputation
        self.should_normalize = perform_normalization
        self.should_perform_pca = perform_pca
        self.should_split_data = split_data

    def impute_missing_values(self, df):
        imputer = SimpleImputer(strategy='most_frequent')
        df['msi_status'] = imputer.fit_transform(df[['msi_status']])
        return df

    def normalize_data(self, df):
        data_for_normalization = df.select_dtypes(include=['float64', 'int64'])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_normalization)
        normalized_df = pd.DataFrame(scaled_data, columns=data_for_normalization.columns)
        return normalized_df

    def perform_pca(self, df):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])
        return pca_df

    def split_and_save_data(self, df):
        # Assuming the principal components are named 'Principal Component 1' and 'Principal Component 2'
        X_pca = df[['Principal Component 1', 'Principal Component 2']].values

        X_train, X_rest = train_test_split(X_pca, test_size=0.4, random_state=42)
        X_val, X_test = train_test_split(X_rest, test_size=0.5, random_state=42)

        train_file = f"{self.output_folder}/train_data.csv"
        val_file = f"{self.output_folder}/validation_data.csv"
        test_file = f"{self.output_folder}/test_data.csv"

        self.write_data_to_file(train_file, X_train, df.columns)
        self.write_data_to_file(val_file, X_val, df.columns)
        self.write_data_to_file(test_file, X_test, df.columns)
    
    def write_data_to_file(self, file_name, data, header):
        with open(file_name, "w") as f:
            f.write(','.join(header) + '\n')
            for sample in data:
                f.write(','.join(map(str, sample)) + '\n')

    def run(self):
        df = None
        if self.should_normalize and self.raw_file_path:
            df = pd.read_csv(self.raw_file_path)
            df = self.normalize_data(df)
            df.to_csv(f'{self.output_folder}/normalized_data.csv', index=False)

        if self.should_impute and self.imputation_file_path:
            imputation_df = pd.read_csv(self.imputation_file_path)
            imputation_df = self.impute_missing_values(imputation_df)
            imputation_df.to_csv(f'{self.output_folder}/imputed_data.csv', index=False)
            # Optionally merge imputed data with the main dataframe as needed
            if df is not None:
                df = df.merge(imputation_df, how='left', on='some_common_column')

        if self.should_perform_pca and df is not None:
            df = self.perform_pca(df)
            df.to_csv(f'{self.output_folder}/pca_data.csv', index=False)
        
        if self.should_split_data and df is not None:
            self.split_and_save_data(df)

preprocessor = DataPreprocessor(raw_file_path='../../data/raw/tcga_rna_count_data_crc.csv',
                                imputation_file_path='../../data/raw/prediction_file_crc.csv',
                                perform_imputation=False,
                                perform_normalization=True,
                                perform_pca=True,
                                split_data=True
)
preprocessor.run()
