import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_path, selected_features_path):
        self.model = joblib.load(model_path)
        # Load selected features
        self.selected_features = pd.read_csv(selected_features_path, header=None).squeeze().tolist()

    def load_data(self, features_path, labels_path):
        # Load the dataset
        X_full = pd.read_csv(features_path)
        y = pd.read_csv(labels_path).squeeze()
        # Subset the features to the selected features
        X = X_full[self.selected_features]
        return X, y

    def evaluate(self, X, y, set_name="Test"):
        # Predictions and metrics
        y_pred = self.model.predict(X)
        print(f"Metrics for {set_name} Set:")
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        print(f"F1 Score (Macro): {f1_score(y, y_pred, average='macro'):.4f}")
        print(f"F1 Score (Micro): {f1_score(y, y_pred, average='micro'):.4f}")
        print(f"F1 Score (Weighted): {f1_score(y, y_pred, average='weighted'):.4f}")

        # Confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize":16})
        plt.xlabel('Predicted Labels', fontsize=18)
        plt.xticks(fontsize=16)
        plt.ylabel('True Labels', fontsize=18)
        plt.yticks(fontsize=16)
        plt.title(f"{set_name} Set Confusion Matrix with 100 Features", fontsize=20, pad=20)
        plt.show()

        # Classification report
        print(f"Classification Report for {set_name} Set:\n", classification_report(y, y_pred))

def main(model_path, selected_features_path, test_features_path, test_labels_path):
    evaluator = ModelEvaluator(model_path, selected_features_path)

    # Load and evaluate test data
    X_test, y_test = evaluator.load_data(test_features_path, test_labels_path)
    print("\nEvaluating Test Set:")
    evaluator.evaluate(X_test, y_test, set_name="Test")

if __name__ == "__main__":
    model_path = 'rf_model_final.joblib'
    selected_features_path = '../../data/final/selected_features.csv' 
    test_features_path = '../../data/final/test_features_80:20.csv'
    test_labels_path = '../../data/final/test_labels_80:20.csv'

    main(model_path, selected_features_path, test_features_path, test_labels_path)
