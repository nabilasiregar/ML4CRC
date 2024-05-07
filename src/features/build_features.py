import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import pdb

model = joblib.load('../../data/jobs/best_random_forest_model_without_pca.joblib')
train_features = pd.read_csv('../../data/final/train_features_smote.csv').columns.tolist() 

# Make sure the columns are in the same order as used in training
pdb.set_trace()
X_train = train_features.loc[:, model.feature_names_in_]
importances = model.feature_importances_

feature_importances = pd.DataFrame({
    'gene': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

top_features = feature_importances.head(200)

print(top_features)

top_features.to_csv("../../data/final/top_200_features.csv", index=False)
