

import pandas as pd
import numpy as np
from gprofiler import GProfiler

# Load the correlation matrix
df = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/correlation_matrix/correlation_matrix.csv", index_col=0)

# Ensure symmetry and matching index/columns
assert df.shape[0] == df.shape[1], "Matrix must be square"
assert all(df.columns == df.index), "Row and column names must match"

# Threshold for poor correlation
threshold = 0.001

# Find proteins where ALL off-diagonal correlations are < threshold (in absolute value)
poor_correlators = []

for protein in df.index:
    correlations = df.loc[protein].drop(protein)  # exclude self-correlation
    if (correlations.abs() < threshold).all():
        poor_correlators.append(protein)

print(f"Found {len(poor_correlators)} poor correlators (|correlation| < {threshold}).")

# Perform GO enrichment using mouse genes
if poor_correlators:
    gp = GProfiler(return_dataframe=True)
    results = gp.profile(organism='mmusculus', query=poor_correlators)

    # Save results
    results.to_csv("go_enrichment_results_mouse.csv", index=False)
    print("GO enrichment results saved to go_enrichment_results_mouse.csv")
else:
    print("No poor correlators found.")
