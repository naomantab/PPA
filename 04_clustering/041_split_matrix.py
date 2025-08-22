import pandas as pd
import os
from collections import defaultdict

# Load the full matrix
matrix = pd.read_csv("/data/home/bt23917/PPA/04_clustering/NormalisedMatrix(Quantile)(Z-score).csv")
dataset_col = matrix['DatasetName']
matrix = matrix.drop(columns=["DatasetName"])

# Group phosphosite columns by protein prefix
prefix_map = defaultdict(list)
for col in matrix.columns:
    prefix = col.split("_")[0]
    prefix_map[prefix].append(col)

# Split proteins into chunks (e.g., 100 proteins each)
prefixes = list(prefix_map.keys())
chunk_size = 100
chunks = [prefixes[i:i + chunk_size] for i in range(0, len(prefixes), chunk_size)]

# Output directory
out_dir = "/data/home/bt23917/PPA/04_clustering/split_matrices"
os.makedirs(out_dir, exist_ok=True)

# Save each chunk
for i, chunk in enumerate(chunks):
    subset_cols = []
    for prefix in chunk:
        subset_cols.extend(prefix_map[prefix])
    subset_df = pd.concat([dataset_col, matrix[subset_cols]], axis=1)
    subset_df.to_csv(os.path.join(out_dir, f"matrix_chunk_{i+1}.csv"), index=False)

print(f"Split into {len(chunks)} files.")
