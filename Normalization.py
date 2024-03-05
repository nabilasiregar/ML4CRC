import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = '/Users/bryanlee/Desktop/processing_data/tcga_rna_count_data_crc.csv'
rna_data = pd.read_csv(file_path)
#keep gene names
rna_counts = rna_data.set_index(rna_data.columns[0])

# Calculate TPM
# 1 : RPK (Reads Per Kilobase)
gene_lengths = [1000] * len(rna_counts)  #assume gene length 1000 bps
RPK = rna_counts.div(gene_lengths, axis=0)

# 2: TPM
sum_RPK = RPK.sum(axis=0)  # Summing over columns not rows
TPM = RPK.div(sum_RPK, axis=1) * 1e6

#outputs .csv file
output_file_path = '/Users/bryanlee/Desktop/processing_data/normalized_tcga_rna_tpm.csv'
TPM.to_csv(output_file_path)

#Z-Scores from TPM. ALL genes mean = 0, SD = 1
#Useful to compare expression levels across (between) samples and is suitable  for SVM, logistic regression, neural networks.

#remove first column = only numerical data
rna_counts = rna_data.drop(rna_data.columns[0], axis=1)

#assume gene length 1000 bps
gene_lengths = [1000] * len(rna_counts)
RPK = rna_counts.div(gene_lengths, axis=0)

# Calculate TPM (Transcripts Per Million)
sum_RPK = RPK.sum()
TPM = RPK.div(sum_RPK, axis=1) * 1e6

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform (and transposes features to row for StandardScaler)
TPM_scaled = scaler.fit_transform(TPM.T)

# scaled data to dataframe
TPM_scaled_df = pd.DataFrame(TPM_scaled, index=TPM.columns, columns=TPM.index)

#outputs .csv file
output_file_path = '/Users/bryanlee/Desktop/processing_data/Z_scores_normalized_tcga_rna_tpm.csv'
TPM_scaled_df.to_csv(output_file_path)