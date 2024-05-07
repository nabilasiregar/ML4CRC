import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import os
import pdb

# Load your dataset
X_train = pd.read_csv('../../data/final/train_features_80:20_smote.csv')
y_train = pd.read_csv('../../data/final/train_labels_80:20_smote.csv').squeeze()

# List of features in order of importance or any other criterion
relevant_genes = [
    'ATOH1', 'GRM8', 'PCDH8', 'CEL', 'CFTR', 'ENGASE', 'GP2', 'C5orf52', 
    'SATB2', 'CDX2', 'CYP2B6', 'NKD1', 'SMAD2', 'MTOR', 'NFE2L2', 'RB1', 
    'KEAP1', 'TERT', 'RASA1', 'CDC73', 'CTNNA1', 'ERBB4', 'CD8A', 'PRF1', 
    'GZMA', 'GZMB', 'CX3CL1', 'CXCL9', 'CXCL10', 'IFNG', 'IL1B', 'LAG3', 
    'CTLA4', 'CD274', 'PDCD1', 'TIGIT', 'IDO1', 'PDCD1LG2', 'VEGFA', 'VEGFB', 'VEGFC', 'VEGFD'
]

# All potential features excluding the known relevant genes
potential_features = [col for col in X_train.columns if col not in relevant_genes]

# Calculate mutual information
mi_scores = mutual_info_classif(X_train[potential_features], y_train)
mi_features = pd.Series(mi_scores, index=potential_features).sort_values(ascending=False)

# Start with the known relevant genes
selected_features = list(relevant_genes)

# Initialize your classifier with specified hyperparameters
clf = RandomForestClassifier(
    n_estimators=130,
    max_depth=21,
    min_samples_split=4,
    min_samples_leaf=3,
    bootstrap=True,
    random_state=42
)

# Define your cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

csv_file_path = 'feature_selection_with_mi.csv'

if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['Num_Features', 'Feature_Set', 'F1_Score', 'Added_Feature']).to_csv(
        csv_file_path, mode='w', index=False
    )

# Add features based on mutual information ranking
for feature in mi_features.index:
    # Add the next feature based on MI ranking
    selected_features.append(feature)
    X_train_selected = X_train[selected_features]

    # Perform cross-validation
    pdb.set_trace()
    f1_scores = cross_val_score(
        clf, X_train_selected, y_train, cv=cv_strategy, 
        scoring=make_scorer(f1_score, average='weighted')
    )
    
    # Prepare data to be appended
    data_to_append = {
        'Num_Features': len(selected_features),
        'Feature_Set': ', '.join(selected_features),
        'F1_Score': np.mean(f1_scores),
        'Added_Feature': feature
    }
    
    # Append results to the CSV file
    pd.DataFrame([data_to_append]).to_csv(csv_file_path, mode='a', header=not pd.read_csv(csv_file_path).size, index=False)

    print(f"Appended results for {feature} to '{csv_file_path}'")