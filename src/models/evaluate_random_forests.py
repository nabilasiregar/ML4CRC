import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        print(f"F1 Score (Macro Average): {f1_score(y_test, y_pred, average='macro')}")
        print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    test_data = pd.read_csv('../../data/processed/test_data.csv')

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1].astype(str)

    evaluator = ModelEvaluator('random_forest_model.joblib')
    evaluator.evaluate(X_test, y_test)
