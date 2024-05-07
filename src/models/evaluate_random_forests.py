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
        plt.figure(figsize=(12, 8))
        heatmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 16})
        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=16)
        plt.xlabel('Predicted Labels', fontsize=18)
        plt.xticks(fontsize=16)
        plt.ylabel('True Labels',  fontsize=18)
        plt.yticks(fontsize=16)
        plt.title(f"{set_name} Set Confusion Matrix for Random Forest Without Feature Selection", fontsize=20, pad=20)
        plt.show()
        
        # Print classification report
        print(f"Classification Report for {set_name} Set:\n", classification_report(y, y_pred))

def main(model_path, train_features_path, train_labels_path, test_features_path, test_labels_path):
    evaluator = ModelEvaluator(model_path)

    # Load and evaluate training data
    # X_train, y_train = evaluator.load_data(train_features_path, train_labels_path)
    # print("Evaluating Training Set:")
    # evaluator.evaluate(X_train, y_train, set_name="Training")

    # Load and evaluate test data
    X_test, y_test = evaluator.load_data(test_features_path, test_labels_path)
    print("\nEvaluating Test Set:")
    evaluator.evaluate(X_test, y_test, set_name="Test")

if __name__ == "__main__":
    output_dir = '../../data/final/'
    model_path = '../../data/jobs/all_features_model.joblib'
    train_features_path = f"{output_dir}/train_features_80:20_smote.csv"
    train_labels_path = f"{output_dir}/train_labels_80:20_smote.csv"
    test_features_path = f"{output_dir}/test_features_80:20.csv"
    test_labels_path = f"{output_dir}/test_labels_80:20.csv"

    main(model_path, train_features_path, train_labels_path, test_features_path, test_labels_path)
