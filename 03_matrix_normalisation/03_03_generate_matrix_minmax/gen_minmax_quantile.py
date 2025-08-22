#!/bin/python

# ----------------- #
# LOAD DEPENDENCIES
# ----------------- #

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re

# ----------------- #
# IMPORT R-NORMALISED MATRIX 
# ----------------- #

# matrix = pd.read_csv('/data/home/bty449/ExplainableAI/MatrixCSVs/NBA-Matrix_Quantile.csv', header = 0)
matrix = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NBA-Matrix_Quantile.csv', header = 0)
# matrix = pd.read_csv('/data/home/bt23917/PPA/RawMatrix_NoOutliers.csv', header = 0)

matrix = matrix.set_index('DatasetName')

# ----------------- #
# ENSURE OUTPUT DIRECTORY EXISTS
# ----------------- #
os.makedirs('./ScaledMatrix', exist_ok=True)

print(os.getcwd())


# ----------------- #
# START LOOPING THROUGH DATASETS
# ----------------- #
for dataset in matrix.index:
    # Get the row for the current dataset
    row = matrix.loc[dataset]
    
    # Convert the row to a numpy array and reshape it
    row_array = np.array(row).reshape(-1, 1)
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fit and transform the data
    scaled_row = scaler.fit_transform(row_array)
    
    # Convert back to a DataFrame
    scaled_row_df = pd.DataFrame(scaled_row, index=row.index, columns=[dataset])
    
    # Save the scaled row to a CSV file
    safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in dataset)
    scaled_row_df.to_csv(f'./ScaledMatrix/{safe_name}_scaled.csv')


# ----------------- #
# READ BACK ALL SCALED FILES AND PRINT HEAD
# ----------------- #
for file in os.listdir('./ScaledMatrix/'):
    if file.endswith('_scaled.csv'):
        # Read the scaled CSV file
        df = pd.read_csv(f'./ScaledMatrix/{file}', index_col=0)
        
        # Print the first few rows of the DataFrame
        print(f"Contents of {file}:")
        print(df.head())
        print("\n")

# ----------------- #


all_dfs = []

for file in os.listdir('./ScaledMatrix/'):
    if file.endswith('_scaled.csv'):
        # Read the scaled CSV file
        df = pd.read_csv(f'./ScaledMatrix/{file}', index_col=0)
        
        # Append to list
        all_dfs.append(df)

# Combine all DataFrames vertically (stack rows)
combined_df = pd.concat(all_dfs, axis=1)

combined_df  = combined_df.T   


combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)

# fix column names to match the expected format
combined_df.columns = [
    col if col == "DatasetName" else re.sub(r"(.*)\.(\d+)\.$", r"\1(\2)", col)
    for col in combined_df.columns
]

combined_df = combined_df.loc[:, combined_df.count() != 0]

# Save the combined DataFrame to CSV
combined_df.to_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile).csv', index=True, index_label="DatasetName")


# Print the first few rows
print("Combined DataFrame head:")
print(combined_df.head())
