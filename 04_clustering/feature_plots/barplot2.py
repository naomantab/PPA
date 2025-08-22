import pandas as pd
import matplotlib.pyplot as plt


base_dir = 'C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/results'
norm = pd.read_csv(f'{base_dir}/clustered_matrix.csv', header=0).shape[1] - 1
val50 = pd.read_csv(f'{base_dir}/clustered_matrix_min50vals.csv', header=0).shape[1] - 1
val100 = pd.read_csv(f'{base_dir}/clustered_matrix_min100vals.csv', header=0).shape[1] - 1
val150 = pd.read_csv(f'{base_dir}/clustered_matrix_min150vals.csv', header=0).shape[1] - 1
val200 = pd.read_csv(f'{base_dir}/clustered_matrix_min200vals.csv', header=0).shape[1] - 1


# Barplot data
labels = ['0','50', '100', '150', '200']
counts = [norm, val50, val100, val150, val200] 

# Plot
plt.figure(figsize=(6, 5))
bars = plt.bar(labels, counts, color=['blue'], width=0.3)
plt.ylabel('Number of features')
plt.xlabel('Thresholds')
plt.title('Feature counts per dataset type')

# Add count labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 10, str(height), ha='center', va='bottom')

plt.tight_layout()

plt.savefig('C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/features_clustered_plot.png') 
