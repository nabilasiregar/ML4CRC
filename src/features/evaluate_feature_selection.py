import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_path, relevant_features=None):
        self.model = joblib.load(model_path)
        # If relevant_features is not provided, infer from model if possible
        self.relevant_features = relevant_features if relevant_features is not None else self.infer_relevant_features()

    def infer_relevant_features(self):
        """Infer relevant features from the model if possible."""
        # Attempt to infer feature names from the model
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        return None

    def load_data(self, features_path, labels_path):
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path).squeeze()
        # Select only the relevant features if they are known
        if self.relevant_features is not None:
            X = X[self.relevant_features]
        return X, y

    def evaluate(self, X, y, set_name="Test"):
        # Predict using the trained model
        y_pred = self.model.predict(X)
        
        # Calculate and print metrics
        print(f"Metrics for {set_name} Set:")
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        print(f"F1 Score (Macro): {f1_score(y, y_pred, average='macro'):.4f}")
        print(f"F1 Score (Micro): {f1_score(y, y_pred, average='micro'):.4f}")
        print(f"F1 Score (Weighted): {f1_score(y, y_pred, average='weighted'):.4f}")
        
        # Plot confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f"{set_name} Set Confusion Matrix")
        plt.show()
        
        # Print classification report
        print(f"Classification Report for {set_name} Set:\n", classification_report(y, y_pred))

if __name__ == "__main__":
    # Define paths and initialize the evaluator with the model path and relevant features
    model_path = '../../data/jobs/selected_features_model.joblib'
    relevant_features = [
        'ATOH1', 'GRM8', 'PCDH8', 'CEL', 'CFTR', 'ENGASE', 'GP2', 'C5orf52', 
        'SATB2', 'CDX2', 'CYP2B6', 'NKD1', 'SMAD2', 'MTOR', 'NFE2L2', 'RB1', 
        'KEAP1', 'TERT', 'RASA1', 'CDC73', 'CTNNA1', 'ERBB4', 'CD8A', 'PRF1', 
        'GZMA', 'GZMB', 'CX3CL1', 'CXCL9', 'CXCL10', 'IFNG', 'IL1B', 'LAG3', 
        'CTLA4', 'CD274', 'PDCD1', 'TIGIT', 'IDO1', 'PDCD1LG2', 'VEGFA', 'VEGFB', 'VEGFC', 'VEGFD'
    ]
    evaluator = ModelEvaluator(model_path, relevant_features=relevant_features)
    
    # Define paths to your data
    test_features_path = '../../data/final/test_features_80:20.csv'
    test_labels_path = '../../data/final/test_labels_80:20.csv'

    # Load data and evaluate
    X_test, y_test = evaluator.load_data(test_features_path, test_labels_path)
    evaluator.evaluate(X_test, y_test, set_name="Test")
