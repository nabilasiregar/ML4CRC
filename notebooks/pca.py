import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data from the specified path
file_path = r'C:\Users\eirin\OneDrive\Υπολογιστής\normalized_tcga_rna_tpm.csv'
df = pd.read_csv(file_path)

# Assuming the first column is an index or identifiers, let's use all other columns for PCA
df_for_pca = df.iloc[:, 1:]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_for_pca) 

# Initialize PCA, reducing the data to 2 dimensions for visualization
pca = PCA(n_components=2)

# Fit and transform the scaled data using PCA
pca_result = pca.fit_transform(scaled_data)
print(pca_result)

# Convert the PCA result to a DataFrame for easier plotting
pca_df = pd.DataFrame(data = pca_result, columns = ['Principal Component 1', 'Principal Component 2'])



# Plotting the PCA results
plt.figure(figsize=(8,6))
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Gene Count Data')
plt.show()

reduced_data_path = r'C:\Users\eirin\OneDrive\Υπολογιστής\reduced_tcga_rna_tpm.csv'
pca_df.to_csv(reduced_data_path, index=False)


