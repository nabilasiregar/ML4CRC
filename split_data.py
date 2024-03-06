import pandas as pd
from sklearn.model_selection import train_test_split

# File path of the original data
file_path = r'C:\Users\angel\Desktop\machine-learning-master\reduced_tcga_rna_tpm.csv'

# Load the original data into a DataFrame
df = pd.read_csv(file_path)

# Assuming your principal component columns are named 'Principal Component 1' and 'Principal Component 2'
X_pca = df[['Principal Component 1', 'Principal Component 2']].values

# Splitting the data into 60% for training and 40% for the rest
X_train, X_rest = train_test_split(X_pca, test_size=0.4, random_state=42)

# Splitting the rest (40%) into 50% for validation and 50% for testing
X_val, X_test = train_test_split(X_rest, test_size=0.5, random_state=42)

# Save each split into separate files
train_file = "train_data.csv"
val_file = "validation_data.csv"
test_file = "test_data.csv"

# Function to write data to file
def write_data_to_file(file_name, data, header):
    with open(file_name, "w") as f:
        if header is not None:
            f.write(','.join(map(str, header)) + '\n')  # Write header
        for sample in data:
            f.write(','.join(map(str, sample)) + '\n')  # Write data

# Save training data with header
write_data_to_file(train_file, X_train, df.columns)

# Save validation data with header
write_data_to_file(val_file, X_val, df.columns)

# Save testing data with header
write_data_to_file(test_file, X_test, df.columns)
