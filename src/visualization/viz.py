import pandas as pd
import matplotlib.pyplot as plt

# Load the results from CSV
df_results = pd.read_csv('feature_selection_f1_scores.csv')

# Plotting the F1 Score as a function of the number of features used
plt.figure(figsize=(10, 6))
plt.plot(df_results['Num_Features'], df_results['F1_Score'], marker='o', linestyle='-', color='b')
plt.title('F1 Score by Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.grid(True)
plt.show()
