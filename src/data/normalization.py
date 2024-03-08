import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataNormalizer:
    def __init__(self, file_path, output_path_tpm, output_path_z_scores):
        self.file_path = file_path
        self.output_path_tpm = output_path_tpm
        self.output_path_z_scores = output_path_z_scores

    def calculate_TPM(self):
        rna_data = pd.read_csv(self.file_path)
        rna_counts = rna_data.set_index(rna_data.columns[0])
        gene_lengths = [1000] * len(rna_counts)  # Assume gene length 1000 bps
        RPK = rna_counts.div(gene_lengths, axis=0)
        sum_RPK = RPK.sum(axis=0)
        TPM = RPK.div(sum_RPK, axis=1) * 1e6
        TPM.to_csv(self.output_path_tpm)
        return TPM

    def normalize_data(self):
        TPM = self.calculate_TPM()
        rna_counts = pd.read_csv(self.file_path).drop(rna_data.columns[0], axis=1)
        scaler = StandardScaler()
        TPM_scaled = scaler.fit_transform(TPM.T)
        TPM_scaled_df = pd.DataFrame(TPM_scaled, index=TPM.columns, columns=TPM.index)
        TPM_scaled_df.to_csv(self.output_path_z_scores)
