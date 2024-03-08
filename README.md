# Data pre-processsing

## How to use
### Imputation
```
imputer = DataImputer(filepath='../data/raw/prediction_file_crc.csv',output_path='../data/processed/prediction_file_crc_imputed.csv')
imputer.impute_missing_values()
```

### Normalization
```
normalizer = DataNormalizer(file_path='../data/raw/tcga_rna_count_data_crc.csv',output_path_tpm='../data/processed/normalized_tcga_rna_tpm.csv', output_path_z_scores='../data/processed/Z_scores_normalized_tcga_rna_tpm.csv')
normalizer.calculate_TPM()  # save the TPM data to a CSV file
normalizer.normalize_data()  # save the Z-score normalized data to a CSV file
```

### PCA Analysis
```
pca_analyzer = PCAAnalysis(file_path='normalized_tcga_rna_tpm.csv', reduced_data_path='reduced_tcga_rna_tpm.csv')
pca_analyzer.perform_pca()
```
