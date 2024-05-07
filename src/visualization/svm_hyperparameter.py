import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('svm_log.csv')

df.columns = ['trial', 'C_value', 'gamma_value', 'kernel_value', 'f1_score']

# Create a new column for the grouped trial numbers
df['trial_group'] = df['trial']
max_f1_row = df.loc[df['f1_score'].idxmax()]
max_f1_trial = max_f1_row['trial']
max_f1_value = max_f1_row['f1_score']
# Plotting
plt.figure(figsize=(10, 6))

# Blue dots representing each trial's objective value
plt.scatter(df['trial'], df['f1_score'], alpha=0.6, label='Objective Value', color='blue')

# Highlight the dot with the highest F1 score in green
plt.scatter(max_f1_row['trial'], max_f1_row['f1_score'], color='green', s=50, edgecolors='black', label='Highest F1 Score')
plt.text(max_f1_trial, max_f1_value, f'  Trial {max_f1_trial}', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=14)

# Red horizontal line representing the best value (which is 1 in this case)
plt.axhline(y=1, color='red', linestyle='-', linewidth=2, label='Best Value')

# Add a vertical line to indicate the trial with the highest F1 score
plt.axvline(x=max_f1_row['trial'], color='green', linestyle='--', linewidth=2, label='Best Trial')

# Set the title and labels of the plot
plt.title('SVM Hyperparameters Optimization', fontsize=20)
plt.xlabel('Trial', fontsize=18)
plt.xticks(fontsize=16)
plt.ylabel('F1 Score', fontsize=18)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)

plt.show()
