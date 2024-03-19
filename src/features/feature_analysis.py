import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Important Features")
    plt.barh(range(top_n), importances[indices], align="center")
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()

if __name__ == "__main__":
    model_path = "../models/best_random_forest_model_with_smote_rna_only.joblib"
    model = joblib.load(model_path)

    data_path = "../../data/processed/train_data_oversampled.csv"
    data_for_features = pd.read_csv(data_path)
    feature_names = data_for_features.columns[:-1]  # Exclude the target variable

    plot_feature_importance(model, feature_names, top_n=20)
