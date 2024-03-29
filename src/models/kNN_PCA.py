import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def load_data(features_file, target_file):
    features = pd.read_csv(features_file)
    target = pd.read_csv(target_file)
    return features, target

def encode_target(target):
    target_mapping = {"MSS": 0, "MSI-H": 1, "MSI-L": 2, np.nan: 3, 2: 2}
    return target.apply(lambda x: target_mapping.get(x, -1))  # Use get method to handle missing keys

def knn_classification(features_train, target_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(features_train, target_train)
    return knn

def calculate_accuracy(model, features, target):
    y_pred = model.predict(features)
    return accuracy_score(target, y_pred)

# File paths
features_file = '/ML4CRC/src/data/resampled_data.csv'
target_file = '/ML4CRC/src/data/resampled_target.csv'
test_data_file = '/ML4CRC/src/data/test_data.csv'
validation_file = '/ML4CRC/src/data/val_data.csv'

# Load data
features, target = load_data(features_file, target_file)
test_data = pd.read_csv(test_data_file)
validation_data = pd.read_csv(validation_file)

# Encode target labels
target = encode_target(target['msi_status'])
test_target = encode_target(test_data['msi_status'])
validation_target = encode_target(validation_data['msi_status'])

# Drop target column from features
features.drop(columns=['msi_status'], inplace=True)
test_data.drop(columns=['msi_status'], inplace=True)
validation_data.drop(columns=['msi_status'], inplace=True)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.95)  # Choose number of components to explain 95% of variance
features_pca = pca.fit_transform(features)
test_data_pca = pca.transform(test_data)
validation_data_pca = pca.transform(validation_data)

# Train KNN model
knn_model = knn_classification(features_pca, target)

# Calculate accuracy using cross-validation
cv_scores = cross_val_score(knn_model, features_pca, target, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))

# Calculate accuracy on test data
test_accuracy = calculate_accuracy(knn_model, test_data_pca, test_target)
print("Accuracy on test data:", test_accuracy)

# Calculate accuracy on validation data
validation_accuracy = calculate_accuracy(knn_model, validation_data_pca, validation_target)
print("Accuracy on validation data:", validation_accuracy)
