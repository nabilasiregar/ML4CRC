import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_path, columns_to_exclude=None):
        self.model = joblib.load(model_path)
        self.columns_to_exclude = columns_to_exclude

    def load_and_prepare_data(self, file_path):
        data = pd.read_csv(file_path)
        if self.columns_to_exclude:
            data = data.drop(columns=self.columns_to_exclude)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].astype(str)
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

def evaluate_model(train_data_path, test_data_path, model_path, columns_to_exclude=None):
    evaluator = ModelEvaluator(model_path, columns_to_exclude)
    X_train, y_train = evaluator.load_and_prepare_data(train_data_path)
    train_metrics = evaluator.evaluate(X_train, y_train, set_name="Training", show_classification_report=False)
    
    X_test, y_test = evaluator.load_and_prepare_data(test_data_path)
    test_metrics = evaluator.evaluate(X_test, y_test, set_name="Test", show_classification_report=True)

if __name__ == "__main__":
    train_data_path = '../../data/processed/train_data_oversampled.csv'
    test_data_path = '../../data/processed/test_data.csv'
    model_path = 'best_random_forest_model_oversampled.joblib'
    evaluate_model(train_data_path, test_data_path, model_path, columns_to_exclude=None)
