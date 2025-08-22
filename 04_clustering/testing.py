import pandas as pd

# Load the dataset
data = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/results/clustered_matrix.csv')

# Create a list of proteins by splitting the column names at the underscore and getting the first part
proteins = [col.split('_')[0] for col in data.columns]

# Get the unique proteins that start with 'P'
p_proteins = [protein for protein in set(proteins) if protein.startswith('PIK3')]

# Print the result
print(p_proteins)
