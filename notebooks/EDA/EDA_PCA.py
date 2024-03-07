

####Visualize and analyze PCA data###


import pandas as pd
from scipy.stats import pointbiserialr

# Load your gene expression data and target feature into a DataFrame
gene_expression_data = pd.read_csv('/home/rodrigo/Documents/Bioinformatics_&_SB/S-ML/ML4CRC/data/z-core_normalized_tcga_rna_count_data_crc.csv')
#gene_expression_data = gene_expression_data.transpose()
gene_expression_data = gene_expression_data.iloc[0: , 1:]
print(gene_expression_data.shape)
target = pd.read_csv('/home/rodrigo/Documents/Bioinformatics_&_SB/S-ML/ML4CRC/data/prediction_file_crc_imputed.csv')
target_feature_data = target['msi_status']
print(target_feature_data.shape)
## Mapping categories to numerical values
category_mapping = {'MSS': 0, 'Indeterminate': 1, 'MSI-L': 2, 'MSI-H': 3}

## Assuming numerical gene expression data is in columns Gene1, Gene2, ..., and the target feature is in column 'Target'
# Calculate the Point-Biserial Correlation Coefficient between each gene and the target feature
correlation_results = {}
for gene_column in gene_expression_data.columns:
    correlation_coef, p_value = pointbiserialr(gene_expression_data[gene_column], target_feature_data.map(category_mapping))
    correlation_results[gene_column] = correlation_coef

# Convert results to DataFrame for easier manipulation
correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index', columns=['Correlation'])

# Get the top 10 highest correlations
top_10_correlations = correlation_df.nlargest(10, 'Correlation')

# Print the top 10 highest correlations
print(top_10_correlations)