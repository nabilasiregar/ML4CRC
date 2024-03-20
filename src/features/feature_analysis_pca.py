import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model_path = '../../data/final/best_random_forest_model.joblib'
model = joblib.load(model_path)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [f'PC{i+1}' for i in range(len(importances))]
N = 10

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(N), importances[indices][:N], color="r", align="center")
plt.xticks(range(N), [feature_names[i] for i in indices[:N]], rotation=45)
plt.xlim([-1, N])
plt.ylabel('Relative Importance')
plt.tight_layout()
plt.show()

print("Top 10 Feature importances:")
for i in range(N):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

selected_features = [feature_names[i] for i in indices[:N]]

cumulative_importance_threshold = 0.95
cumulative_importances = np.cumsum(importances[indices])
threshold_index = np.where(cumulative_importances > cumulative_importance_threshold)[0][0] + 1
selected_features_threshold = [feature_names[i] for i in indices[:threshold_index]]

print("Selected features based on fixed N:", selected_features)
print("Selected features based on threshold:", selected_features_threshold)
