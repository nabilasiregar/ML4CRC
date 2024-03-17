import pandas as pd
from imblearn.over_sampling import SMOTE

file_path = '../../data/processed/train_data_impute.csv'
train_data = pd.read_csv(file_path)

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

oversampled_train_data = pd.concat([X_res, y_res], axis=1)

oversampled_file_path = '../../data/processed/train_data_oversampled_imputation.csv'
oversampled_train_data.to_csv(oversampled_file_path, index=False)
