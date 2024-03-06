

### Imputation but in python file


import pandas as pd

# Read the raw data
raw_data = pd.read_csv('/home/rodrigo/Documents/Bioinformatics_&_SB/S-ML/ML4CRC/data/raw/prediction_file_crc.csv')

# Display value counts of 'msi_status' before imputation
print("Value counts before imputation:")
print(raw_data['msi_status'].value_counts())

# Impute missing values using the mode (most frequent value)
mode_value = raw_data['msi_status'].mode()[0]
raw_data['msi_status'] = raw_data['msi_status'].fillna(mode_value)

# Display value counts of 'msi_status' after imputation
print("\nValue counts after imputation:")
print(raw_data['msi_status'].value_counts())

# Save the processed data to a new CSV file
raw_data.to_csv('/home/rodrigo/Documents/Bioinformatics_&_SB/S-ML/ML4CRC/data/prediction_file_crc_imputed.csv', index=False)





