#!/bin/python


# ---------------- #
# LOAD DEPENDENCIES
# ---------------- #

print('About to import dependencies')
import sys
print('Sys loaded')
import os
print('OS loaded')
import pandas as pd
print('Pandas loaded')
import numpy as np
print('Numpy loaded')

grandparent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(grandparent_dir)
from funcs import preprocessing

print('Functions imported')

#-------------------------- #
# GENERATE RAW MATRIX HEADER
#-------------------------- #

# load preprocessed datasets
# stores names of processed datasets

file_names = preprocessing.get_csv_file_names_as_tuple(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets')
print('File names loaded')

files_dict = preprocessing.create_dict_per_dataset(file_names)
print(files_dict)

matrix_cols = preprocessing.create_matrix_header(files_dict)
print(f'Matrix header:', matrix_cols)

#-------------------- #
# LOAD MATRIX HEADER
#-------------------- #

matrix = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/raw-matrix-header.csv', header = 0)

# ----------------------------------------------------- #
# LOAD DATASET NAMES & PAIR DATASET NAMES WITH FILENAMES
# ----------------------------------------------------- #

# gets all column names in the format (PY2022_4hrIntensity)
def file_colnames(all_files):
    result = []
    for x in all_files:
        df = pd.read_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{x}.csv', nrows=0)  # Only reads the header
        columns = df.columns[1:]  # Skip first column
        result.append((x,columns))
        print(x)
    return result
    
# files_datasets = file_colnames(all_files_in_folder)
files_datasets = file_colnames(file_names)

print("these are the all column names done")

intermed_matrix = preprocessing.add_rows_to_matrix(matrix, files_datasets, files_dict)
print(intermed_matrix)

# ----------------- #
# FORMAT MATRIX
# ----------------- #

# reorder matrix columns
cols = intermed_matrix.columns.tolist()
cols = cols[-2:-1] + cols[:-2]
raw_matrix = intermed_matrix[cols]
    
# convert only numeric columns (skip 'DatasetName')
numeric_cols = [col for col in raw_matrix.columns if col != 'DatasetName']
for col in numeric_cols:
    raw_matrix[col] = pd.to_numeric(raw_matrix[col], errors='coerce')

# remove infinity values
raw_matrix = raw_matrix.replace([np.inf, -np.inf], np.nan)

# ensure 'DatasetName' is the first column
cols = raw_matrix.columns.tolist()
if 'DatasetName' in cols:
    cols = ['DatasetName'] + [col for col in cols if col != 'DatasetName']
    raw_matrix = raw_matrix[cols]

# save raw matrix
raw_matrix.to_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/RawMatrix.csv', index=False)
print(f'Raw matrix saved successfully!', raw_matrix)


# ----------------- #
