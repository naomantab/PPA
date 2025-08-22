#!/bin/python


# ----------------- #
# LOAD DEPENDENCIES
# ----------------- #

import sys
import os
import pandas as pd
import numpy as np
import requests

grandparent_dir = os.path.abspath(os.getcwd())
sys.path.append(grandparent_dir)
from funcs import preprocessing

# ----------------- #
# LOAD & CLEAN DATA
# ----------------- #

dataset = 'MS2024'

print('Loading raw data for', dataset, '...')
file_name = "1-s2.0-S1535947624001099-mmc2.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name="STable2", header=1)
print('Raw data loaded.')

data = data[data['Localization prob'] >= 0.85] # filter localization probability


columns_saved = [
    "Gene names",
    "Amino acid",
    "Position", 
    "Ratio mod/base",
    "Ratio mod/base Adult-2",
    "Ratio mod/base Adult-3",
    "Ratio mod/base Adult-4",
    "Ratio mod/base Adult-5",
    "Ratio mod/base Adult-6",
    "Ratio mod/base MidAge-1",
    "Ratio mod/base MidAge-2",
    "Ratio mod/base MidAge-3",
    "Ratio mod/base Old-2",
    "Ratio mod/base Old-3",
    "Ratio mod/base Old-4",
    "Ratio mod/base Old-5",
    "Ratio mod/base Old-6"
]

data = data[columns_saved].copy()

# Clean float positions
data = data[np.isfinite(data['Position'])]  # Removes NaN, inf, -inf
data['Position'] = data['Position'].round().astype('Int64')

# create phosphosite column
data['Phosphosite'] = data['Amino acid'].astype(str) + '(' + data['Position'].astype(str) + ')'
# rename GeneName column and remove blanks
data.rename(columns={'Gene names': 'GeneName'}, inplace=True)
data = data.dropna(subset=['GeneName'])

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# cleaning up the dataframe final time
# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]
# drop columns


data.drop('Amino acid', axis=1, inplace=True)
data.drop('Position', axis=1, inplace=True)

# remove columns where all cells are empty
data = data.dropna(subset=data.columns[1:], how='all')

# drop rows where it contains non-conformant char
data = data[~data['phosphosite_ID'].str.contains((';|:|-'))]

# capitalise the first col
data['phosphosite_ID'] = data['phosphosite_ID'].str.upper()

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# log the data
data= preprocessing.log2_transform(data)

data = preprocessing.clean_phosID_col(data) # call function to clean phosphosite_ID column
print('Phosphosite IDs cleaned.')

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)