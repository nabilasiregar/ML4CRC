



import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data from the specified path
file_path = r'/home/rodrigo/Documents/Bioinformatics_&_SB/S-ML/ML4CRC/data/normalized_tcga_rna_count_data_crc.csv'
df = pd.read_csv(file_path)

# Transpose the DataFrame so that genes are treated as samples
df_transposed = df.transpose()

# Assuming the first column is an index or identifiers, let's use all other columns for PCA
df_for_pca = df_transposed.iloc[1:]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_for_pca)

# Initialize PCA, reducing the data to 2 dimensions for visualization
pca = PCA(n_components=2)

# Fit and transform the scaled data using PCA
pca_result = pca.fit_transform(scaled_data)

# Convert the PCA result to a DataFrame for easier plotting
pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])

# Insert gene names back into the DataFrame
pca_df.insert(0, 'Samples', df_transposed.index[1:])

# Plotting the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Gene Count Data')
plt.show()

# Save the reduced data to a CSV file
reduced_data_path = r'/home/rodrigo/Documents/Bioinformatics_&_SB/S-ML/ML4CRC/data/gene_reduced_tcga_rna_count_data_crc.csv'
pca_df.to_csv(reduced_data_path, index=False)
