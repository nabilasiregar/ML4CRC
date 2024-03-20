import matplotlib.pyplot as plt
import pandas as pd

feature_importances_path = '../../data/final/top_features_without_pca.csv'
feature_importances = pd.read_csv(feature_importances_path)

# Assuming the CSV has columns 'Feature' for the gene names and 'Importance' for their importance scores
# and that the data is already sorted by importance
top_features = feature_importances.head(20)

# Create a bar chart
plt.figure(figsize=(10, 8))
plt.barh(top_features['gene'][::-1], top_features['importance'][::-1], color='skyblue')
plt.xlabel('Importance Score')
plt.title('Top 20 Most Influential Genes')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.tight_layout()  # Adjust the layout to fit everything nicely
plt.show()

