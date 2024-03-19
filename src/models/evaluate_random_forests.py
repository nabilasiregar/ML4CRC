import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_path, selected_features=None):
        self.model = joblib.load(model_path)
        self.selected_features = selected_features

    def load_and_prepare_data(self, file_path):
        data = pd.read_csv(file_path)
        if self.selected_features is not None:
            X = data[self.selected_features]
        else:
            X = data.iloc[:, :-1]  # If no selected features, assume last column is the target
        y = data.iloc[:, -1].astype(str)  # Assume the last column is the target
        return X, y

    def evaluate(self, X, y, set_name="Test", show_classification_report=True):
        y_pred = self.model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'f1_micro': f1_score(y, y_pred, average='micro'),
            'f1_weighted': f1_score(y, y_pred, average='weighted')
        }

        print(f"{set_name} Set Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.capitalize()}: {metric_value:.4f}")

        if show_classification_report:
            conf_matrix = confusion_matrix(y, y_pred)
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'{set_name} Set Confusion Matrix')
            plt.show()
            print(f"Classification Report for {set_name} Set:\n", classification_report(y, y_pred))
        return metrics

def load_selected_features(features_path):
    selected_features = pd.read_csv(features_path, header=None).squeeze()
    if selected_features.iloc[0] == '0':
        selected_features = selected_features[1:]
    return selected_features.tolist()

def evaluate_model(train_data_path, test_data_path, model_path, features_path=None):
    selected_features = load_selected_features(features_path) if features_path else None

    evaluator = ModelEvaluator(model_path, selected_features=selected_features)

    X_train, y_train = evaluator.load_and_prepare_data(train_data_path)
    evaluator.evaluate(X_train, y_train, set_name="Training", show_classification_report=False)

    X_test, y_test = evaluator.load_and_prepare_data(test_data_path)
    evaluator.evaluate(X_test, y_test, set_name="Test", show_classification_report=True)

if __name__ == "__main__":
    train_data_path = '../../data/processed/train_data_oversampled.csv'
    test_data_path = '../../data/processed/test_data.csv'
    model_path = 'best_random_forest_model_oversampled.joblib'
    features_path = None
    evaluate_model(train_data_path, test_data_path, model_path, features_path=features_path)
