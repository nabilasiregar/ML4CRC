import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('../models/') 
from random_forests_model import RandomForestModel

def evaluate_retrained_model(test_features_path, test_labels_path, model_path, scaler_path, pca_path, top_components=5):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_labels_path).squeeze()

    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)[:, :top_components]

    y_pred = model.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score (Macro): {f1_macro:.4f}')

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="Blues")
    plt.title('Confusion Matrix After Feature Selection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print(classification_report(y_test, y_pred))

    return accuracy, f1_macro

test_features_path = '../../data/final/test_features.csv'
test_labels_path = '../../data/final/test_labels.csv'
model_path = '../../data/final/best_random_forest_top5.joblib'
scaler_path = '../../data/final/scaler_model.joblib'
pca_path = '../../data/final/pca_model.joblib'

evaluate_retrained_model(test_features_path, test_labels_path, model_path, scaler_path, pca_path)
