import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, imputation_file_path, normalization_file_path, output_folder):
        self.imputation_file_path = imputation_file_path
        self.normalization_file_path = normalization_file_path
        self.output_folder = output_folder

    def impute_missing_values(self):
        raw_data = pd.read_csv(self.imputation_file_path)
        imputer = SimpleImputer(strategy='most_frequent')
        raw_data['msi_status'] = imputer.fit_transform(raw_data[['msi_status']])
        imputed_file_path = self.output_folder + '/prediction_file_crc_imputed.csv'
        raw_data.to_csv(imputed_file_path, index=False)
        return imputed_file_path

    def normalize_data(self):
        rna_data = pd.read_csv(self.normalization_file_path)
        rna_counts = rna_data.set_index(rna_data.columns[0])
        gene_lengths = [1000] * len(rna_counts)  # Assume gene length 1000 bps
        RPK = rna_counts.div(gene_lengths, axis=0)
        sum_RPK = RPK.sum(axis=0)
        TPM = RPK.div(sum_RPK, axis=1) * 1e6
        scaler = StandardScaler()
        TPM_scaled = scaler.fit_transform(TPM.T)
        TPM_scaled_df = pd.DataFrame(TPM_scaled, index=TPM.columns, columns=TPM.index)
        normalized_file_path = self.output_folder + '/normalized_tcga_rna_tpm.csv'
        TPM_scaled_df.to_csv(normalized_file_path)
        return normalized_file_path

    def perform_pca(self, normalized_file_path):
        df = pd.read_csv(normalized_file_path)
        df_for_pca = df.iloc[:, 1:]  # Exclude gene name/index column
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_for_pca)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])
        reduced_data_path = self.output_folder + '/reduced_tcga_rna_tpm.csv'
        pca_df.to_csv(reduced_data_path, index=False)
        return reduced_data_path

    def split_data(self, pca_data_path):
        df = pd.read_csv(pca_data_path)
        X_pca = df[['Principal Component 1', 'Principal Component 2']].values
        X_train, X_rest = train_test_split(X_pca, test_size=0.4, random_state=42)
        X_val, X_test = train_test_split(X_rest, test_size=0.5, random_state=42)
        self._save_split_data(X_train, "train_data.csv", df.columns)
        self._save_split_data(X_val, "validation_data.csv", df.columns)
        self._save_split_data(X_test, "test_data.csv", df.columns)

    def _save_split_data(self, data, filename, header):
        file_path = self.output_folder + '/' + filename
        data_df = pd.DataFrame(data, columns=header[-2:])  # Assuming last 2 columns are PCs
        data_df.to_csv(file_path, index=False)

    def run(self):
        # Step 1: Imputation
        imputed_file_path = self.impute_missing_values()
        
        # Step 2: Normalization (Assuming this step is applied to a different file)
        normalized_file_path = self.normalize_data()
        
        # Step 3: PCA
        pca_data_path = self.perform_pca(normalized_file_path)
        
        # Step 4: Data Splitting
        self.split_data(pca_data_path)

data_preprocessor = DataPreprocessor('../../data/raw/prediction_file_crc.csv',
                                     '../../data/raw/tcga_rna_count_data_crc.csv',
                                     '../../data/processed')
data_preprocessor.run()
