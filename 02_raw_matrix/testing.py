#!/bin/python

# ----------------- #
# LOAD DEPENDENCIES
# ----------------- #

import sys
import os
import pandas as pd
import numpy as np

grandparent_dir = os.path.abspath(os.getcwd())
sys.path.append(grandparent_dir)
from funcs import preprocessing

# ----------------- #
# DEFINE PI3K/AKT PROTEINS (MOUSE)
# ----------------- #

proteins_of_interest_raw = [
    "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
    "PIK3R1", "PIK3R2",
    "AKT1", "AKT2", "AKT3",
    "PDPK1", "PTEN", "MTOR",
    "TSC1", "TSC2", "RHEB",
    "FOXO1", "FOXO3",
    "GSK3B", "BAD", "IRS1"
]

proteins_of_interest = set(p.upper() for p in proteins_of_interest_raw)

# ----------------- #
# LOAD & CLEAN DATA
# ----------------- #

file_name = "RawMatrix.csv"
data = pd.read_csv(f"C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/{file_name}", index_col=0)
print('Raw data loaded.')
# # Filter the column names that start with "TTN"
# # Extract protein names from the column names
# protein_names = data.columns.str.split("_").str[0]

# # Count the occurrences of each unique protein
# protein_counts = protein_names.value_counts()

# # Identify the protein with the most columns
# most_frequent_protein = protein_counts.idxmax()
# most_frequent_protein_count = protein_counts.max()

# print(f"The protein with the most columns is '{most_frequent_protein}' with {most_frequent_protein_count} columns.")

# # Get the top 3 proteins with the most columns
# top_3_proteins = protein_counts.head(3)

# # Print the top 3 proteins along with their column counts
# print("Top 3 proteins with the most columns:")
# for protein, count in top_3_proteins.items():
#     print(f"Protein: {protein}, Columns: {count}")
print(data.shape)
protein_names = data.columns.str.split("_").str[0]
num_unique_proteins = protein_names.nunique()
print(f'Number of unique proteins: {num_unique_proteins}')


# file_name2 = "RawMatrix_NoOutliers.csv"
# data2 = pd.read_csv(f"C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/{file_name2}", index_col=0)
# print(data2.shape)


# data_cyclo = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile)(Z-score).csv', index_col=0)
# data_cyclo = data_cyclo.apply(pd.to_numeric, errors='coerce')
# data_cyclo_numeric = data_cyclo.select_dtypes(include=[np.number])
# mean_of_means = data_cyclo_numeric.mean().median()
# print(mean_of_means) 



# # Convert all to numeric (NaN if fails)
# data = data.apply(pd.to_numeric, errors='coerce')

# # ----------------- #
# # MATCH PROTEINS TO DATA COLUMNS
# # ----------------- #

# # Extract base protein names (before first underscore) and convert to uppercase
# column_protein_names = {col.split("_")[0].upper() for col in data.columns}

# # Determine which proteins matched and didn't
# matched_proteins = sorted(proteins_of_interest & column_protein_names)
# unmatched_proteins = sorted(proteins_of_interest - column_protein_names)

# # Get full matching columns (with underscores)
# matching_cols = [col for col in data.columns if col.split("_")[0].upper() in matched_proteins]
# matched_data = data[matching_cols]

# # Print results
# print(f"\nMatched {len(matched_proteins)} proteins:")
# print(matched_proteins)

# print(f"\nUnmatched {len(unmatched_proteins)} proteins:")
# print(unmatched_proteins)

# # ----------------- #
# # SAVE MATCHED DATA
# # ----------------- #

# matched_data.to_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/matched_pi3k_akt_protein_columns.csv")
# print("Filtered data saved to matched_pi3k_akt_protein_columns.csv.")

# # Optional: Save matched/unmatched protein lists
# # pd.Series(matched_proteins).to_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/matched_protein_names.csv", index=False)
# # pd.Series(unmatched_proteins).to_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/unmatched_protein_names.csv", index=False)
# print("Matched and unmatched protein lists saved.")

# # ----------------- #
# # OPTIONAL: LARGE NEGATIVE VALUES
# # ----------------- #

# mask = matched_data < -20
# large_values = matched_data[mask].dropna(how='all')
# # large_values.to_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/large_values_pi3k_akt_proteins.csv")
# print("Large negative values in matched PI3K/AKT columns saved.")
