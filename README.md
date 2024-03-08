# Data pre-processsing
Run `python preprocess_data.py` on directory src/data

## How to Use
```
# Scenario 1: Normalize, Perform PCA, but Skip Imputation
preprocessor = DataPreprocessor(raw_file_path='../../data/raw/tcga_rna_count_data_crc.csv',
                                        output_folder='../../data/processed',
                                        perform_imputation=False,
                                        perform_normalization=True,
                                        perform_pca=True,
                                        split_data=True)
preprocessor.run()

# Scenario 2: Normalize, Skip PCA, Impute, and Split
preprocessor = DataPreprocessor(raw_file_path='../../data/raw/tcga_rna_count_data_crc.csv',
                                        imputation_file_path='../../data/raw/prediction_file_crc.csv',
                                        output_folder='../../data/processed',
                                        perform_imputation=True,
                                        perform_normalization=True)
```
