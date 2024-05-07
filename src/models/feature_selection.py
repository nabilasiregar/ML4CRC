import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    train_features_path = '../../data/final/train_features_80:20_smote.csv'
    train_labels_path = '../../data/final/train_labels_80:20_smote.csv'

    # Load the training data
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path).squeeze()

    # Define the relevant genes
    relevant_genes = [
        'ATOH1', 'GRM8', 'PCDH8', 'CEL', 'CFTR', 'ENGASE', 'GP2', 'C5orf52', 
        'SATB2', 'CDX2', 'CYP2B6', 'NKD1', 'SMAD2', 'MTOR', 'NFE2L2', 'RB1', 
        'KEAP1', 'TERT', 'RASA1', 'CDC73', 'CTNNA1', 'ERBB4', 'CD8A', 'PRF1', 
        'GZMA', 'GZMB', 'CX3CL1', 'CXCL9', 'CXCL10', 'IFNG', 'IL1B', 'LAG3', 
        'CTLA4', 'CD274', 'PDCD1', 'TIGIT', 'IDO1', 'PDCD1LG2', 'VEGFA', 'VEGFB', 'VEGFC', 'VEGFD'
    ]

    # Initialize and fit the RandomForestClassifier
    rf_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_estimator.fit(X_train, y_train)

    # Obtain and sort the features by their importance
    feature_importances = pd.Series(rf_estimator.feature_importances_, index=X_train.columns)

    # Exclude already known relevant genes from the list to find new ones
    new_features_to_consider = feature_importances.drop(relevant_genes)

    # Select the top 58 new features based on importance
    additional_features = new_features_to_consider.nlargest(58).index.tolist()

    # Combine the known relevant genes with the additional selected features
    final_selected_features = relevant_genes + additional_features

    # Save the final selected features for later use
    pd.Series(final_selected_features).to_csv('selected_features_100.csv', index=False, header=False)

    print(f"Total selected features: {len(final_selected_features)}")

if __name__ == "__main__":
    main()

