
#### Visualize and analize data###





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
data = pd.read_csv("/home/rodrigo/Documents/Bioinformatics_&_SB/S-ML/ML4CRC/data/gene_reduced_tcga_rna_count_data_crc.csv")

# Step 2: Understand the data
print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Get information about the dataset, including data types and missing values


# Step 5: Visualization
# Histograms for numerical features
data.hist(figsize=(10, 8))
plt.show()

# Box plots for numerical features
sns.boxplot(data=data)
plt.show()

# Pair plot (scatter plot matrix) to visualize pairwise relationships
sns.pairplot(data)
plt.show()

# Heatmap for correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()
