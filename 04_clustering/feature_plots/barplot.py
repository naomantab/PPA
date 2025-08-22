import pandas as pd
import matplotlib.pyplot as plt

# Load phosphosite matrix
matrix = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile)(Z-score).csv', header=0)

# Count phosphosite-resolved features
num_phosphosites = matrix.shape[1] - 1 

# Count protein-resolved features
protein_names = [col.split('_')[0] for col in matrix.columns if col != 'DatasetName']
num_proteins = len(set(protein_names))

# count cluster features
# enter below when aquired
num_clusters = 26359 # until 126c is added

# Barplot data
labels = ['Phosphosites', 'Proteins', 'Clusters']
counts = [num_phosphosites, num_proteins, num_clusters]

# Plot
plt.figure(figsize=(6, 5))
bars = plt.bar(labels, counts, color=['firebrick', 'tomato', 'peachpuff'], width=0.3)
plt.ylabel('Number of features')
plt.title('Feature counts per dataset type')

# Add count labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 10, str(height), ha='center', va='bottom')

plt.tight_layout()

plt.savefig('C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/features_plot.png') 
