import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

model = joblib.load('../../data/jobs/best_random_forest_model_without_pca.joblib')
train_features = pd.read_csv('../../data/final/train_features_smote.csv').columns.tolist() 

# Make sure the columns are in the same order as used in training
X_train = train_features.loc[:, model.feature_names_in_]
importances = model.feature_importances_

feature_importances = pd.DataFrame({
    'gene': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

top_features = feature_importances.head(20)

print(top_features)

top_features.to_csv("../../data/final/top_features_without_pca.csv", index=False)
