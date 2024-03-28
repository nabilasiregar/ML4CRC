import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(features_path, labels_path):
    """Load features and labels from specified file paths."""
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()
    return X, y

def select_features(X, relevant_genes):
    """Select and return columns from X specified in relevant_genes."""
    return X[relevant_genes]

def train_model(X_train, y_train, best_params):
    """Train the RandomForestClassifier on the training set with the given best parameters."""
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """Save the trained model to a given filepath."""
    joblib.dump(model, filepath)

def save_dataset(X, filepath):
    """Save the dataset X to the specified filepath."""
    X.to_csv(filepath, index=False)

def main():
    train_features_path = '../../data/final/train_features_80:20_smote.csv'
    train_labels_path = '../../data/final/train_labels_80:20_smote.csv'
    
    best_params = {
        'n_estimators': 130,
        'max_depth': 21,
        'min_samples_split': 4, 
        'min_samples_leaf': 3, 
        'bootstrap': True,    
       'random_state': 42
    }
    
    X, y = load_data(train_features_path, train_labels_path)
    
    relevant_genes = [
        'ATOH1', 'GRM8', 'PCDH8', 'CEL', 'CFTR', 'ENGASE', 'GP2', 'C5orf52',
        'SATB2', 'CDX2', 'CYP2B6', 'NKD1', 'SMAD2', 'MTOR', 'NFE2L2', 'RB1',
        'KEAP1', 'TERT', 'RASA1', 'CDC73', 'CTNNA1', 'ERBB4', 'CD8A', 'PRF1',
        'GZMA', 'GZMB', 'CX3CL1', 'CXCL9', 'CXCL10', 'IFNG', 'IL1B', 'LAG3',
        'CTLA4', 'CD274', 'PDCD1', 'TIGIT', 'IDO1', 'PDCD1LG2', 'VEGFA', 'VEGFB', 'VEGFC', 'VEGFD'
    ]

    X_selected = select_features(X, relevant_genes)

    reduced_features_path = '../../data/final/reduced_training_features.csv'
    save_dataset(X_selected, reduced_features_path)

    model = train_model(X_selected, y, best_params)

    model_filepath = '../../data/jobs/selected_features_model.joblib'
    save_model(model, model_filepath)

    print(f"Model trained with selected features and saved to {model_filepath}")

if __name__ == '__main__':
    main()

