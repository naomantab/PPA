import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from gprofiler import GProfiler

# Load your protein quantification data
data = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile).csv", index_col=0)

# data = data.T

print(data.columns)

data.columns = [data.columns[0]] + [col.split('_')[0] for col in data.columns[1:]]

data = data.groupby(data.columns, axis=1).mean()

data = data.dropna(axis=1, how='all')
data = data.fillna(0)

print(data)
print(data.columns)

# # Step 1: Calculate Spearman correlation
correlation_matrix, _ = spearmanr(data, axis=0)

print(correlation_matrix)

# Convert the correlation matrix to a DataFrame with appropriate labels
correlation_df = pd.DataFrame(correlation_matrix, index=data.columns, columns=data.columns)

# Output to CSV
correlation_df.to_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/CorrelationMatrix.txt", sep='\t')

print("Spearman correlation matrix saved as CSV.")

# # Handle NaN and infinite values in the correlation matrix
# # Replace NaN and infinite values with 0 (or another value like mean, etc.)
# correlation_matrix = np.nan_to_num(correlation_matrix, nan=0, posinf=0, neginf=0)

# correlation_data = pd.DataFrame(correlation_matrix, index=data.columns, columns=data.columns)

# # Optional: Visualize correlation matrix
# clustermap = sns.clustermap(
#     correlation_data,
#     cmap='coolwarm',
#     center=0,
#     figsize=(10, 10),        # Adjust based on size
#     vmin=-1, vmax=1,         # Spearman correlation range
#     xticklabels=False,       # Optional: turn off to save space
#     yticklabels=False
# )
# clustermap.ax_heatmap.set_title("Spearman Correlation Matrix (Subset)")
# clustermap.savefig('C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/spearman_plot.png', dpi=300)


# print("Correlation matrix shape:", correlation_data.shape)
# print("Correlation matrix done")

































# # # Step 2: Hierarchical clustering (complete linkage)
# linked = linkage(correlation_matrix, method='complete')

# # Plot dendrogram
# plt.figure(figsize=(10, 7))
# dendrogram(linked, labels=data.columns.to_list(), orientation='top', distance_sort='descending')
# plt.title("Hierarchical Clustering Dendrogram with Complete Linkage")
# plt.xlabel("Proteins")
# plt.ylabel("Distance")
# plt.savefig('C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/dendrogram_plot.png', dpi=300) 

# # Step 3: Identify highly correlated proteins (based on mean absolute correlation)
# mean_abs_corr = correlation_data.abs().mean(axis=1)
# threshold = mean_abs_corr.quantile(0.9)
# highly_correlated_proteins = mean_abs_corr[mean_abs_corr > threshold]

# # # Step 4: Perform GO enrichment analysis on highly correlated proteins
# genes = highly_correlated_proteins.index.to_list()
# gp = GProfiler(return_dataframe=True)
# result = gp.profile(organism='mmusculus', query=genes)

# print(result.columns)
# print(result)

# # Separate results by domain (Molecular Function, Biological Process, Cellular Component)
# result_mf = result[result['source'] == 'GO:MF']
# result_bp = result[result['source'] == 'GO:BP']
# result_cc = result[result['source'] == 'GO:CC']

# # Print or save the results
# print("Molecular Function Enrichment:")
# print(result_mf[['name', 'p_value', 'term_size']])

# print("Biological Process Enrichment:")
# print(result_bp[['name', 'p_value', 'term_size']])

# print("Cellular Component Enrichment:")
# print(result_cc[['name', 'p_value', 'term_size']])

