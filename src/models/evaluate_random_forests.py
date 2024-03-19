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

def evaluate_model(test_data_path, model_path, columns_to_exclude=None):
    evaluator = ModelEvaluator(model_path, columns_to_exclude)
    X_test, y_test = evaluator.load_and_prepare_data(test_data_path)
    evaluator.evaluate(X_test, y_test)

if __name__ == "__main__":
    # Evaluate model trained on RNA expression data only
    # evaluate_model(
    #     test_data_path='../../data/processed/test_data.csv',
    #     model_path='best_random_forest_model_with_smote_rna_only.joblib',
    #     columns_to_exclude=['TBL', 'TMB', 'aneuploidy_score', 'fraction_genome_altered']
    # )

    evaluate_model(
        test_data_path='../../data/processed/test_data.csv',
        model_path='best_random_forest_model_with_smote.joblib',
        columns_to_exclude=None
    )

    # For evaluating model trained including additional features, simply adjust the `columns_to_exclude` accordingly
    # or set it to `None` if no columns should be excluded
