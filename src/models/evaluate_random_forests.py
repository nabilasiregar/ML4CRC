import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def load_data(self, features_path, labels_path):
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path).squeeze()
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

def main(model_path, train_features_path, train_labels_path, test_features_path, test_labels_path):
    evaluator = ModelEvaluator(model_path)

    # Load and evaluate training data
    X_train, y_train = evaluator.load_data(train_features_path, train_labels_path)
    print("Evaluating Training Set:")
    evaluator.evaluate(X_train, y_train, set_name="Training")

    # Load and evaluate test data
    X_test, y_test = evaluator.load_data(test_features_path, test_labels_path)
    print("\nEvaluating Test Set:")
    evaluator.evaluate(X_test, y_test, set_name="Test")

if __name__ == "__main__":
    output_dir = '../../data/final/'
    model_path = f"{output_dir}best_random_forest_model.joblib"
    train_features_path = f"{output_dir}/train_features_smote_pca.csv"
    train_labels_path = f"{output_dir}/train_labels_pca.csv"
    test_features_path = f"{output_dir}/test_features_pca.csv"
    test_labels_path = f"{output_dir}/test_labels.csv"

    main(model_path, train_features_path, train_labels_path, test_features_path, test_labels_path)
